"""[grokking/whittle] w2: whittle to the floor in frequencies. Adds the move w1's walk lacked (remove a whole
frequency channel at once, with a long retrain) and makes biases first-class prunable groups.

w1 (`whittle.py`) is untouched and still reproduces; everything here is new and imports from it.

MOVES
  channel  remove every group that reads or writes one minted symbol t >= 1 (minted layer-0 / U / V columns of t,
           minted head / W rows of t, and t's head-bias group), cascade, retrain with the LONG budget (fresh
           optimizer, cap C_RETRAIN = 20,000 epochs, early stop once train agreement has held C_SETTLE epochs), gate.
           Channels are offered in ascending total L2 norm (endogenous); the first that passes is admitted; a failed
           channel is restored and the next is tried; the phase ends when every alive channel has failed from the
           current state (or one channel is left).
  group    w1's move (`whittle.whittle_walk`, per-layer round robin, halving fractions), now over the weight layers
           AND the bias layers: per unit for layer-0 / layer-1 biases, per symbol for the minted head bias (the whole
           head bias is on the table). Budget as in w1: cap 500, settle 100.
  The walk alternates phases: channel phase to exhaustion, then group phase to its floor; if the group phase
  admitted anything, the channel phase runs again; it stops when a full cycle admits nothing.
  Gate (unchanged): 0 of 2,822 train pairs disagree with the original net's train argmax. Held-out is logged only.

CASCADE (w2): as w1, except a constant unit's fold into a pruned bias entry revives that entry (per group for the
  minted head bias). The fold removes the unit's bias and >= k outgoing weights and revives <= k bias entries, so it
  never raises either count, and it keeps the function exact.

ARMS (wd 0, seed 42)
  bil_channel          w1's bilinear_both final (3 channels x 3 product units, 54 weights), channel move on, then the
                       group walk (weights + biases)
  bil_channel_lr1e-2   the same at lr 1e-2 (defined; not run: growth was not slow on the bilinear)
  bil_plant1_s<t>      w1's bilinear final with only symbol t's 3 units kept, everything else zero incl. every bias;
                       one long retrain, gate; then the group walk
  bil_from_start       the banked 52-unit bilinear (w1's bilinear_both start), channel + group moves from the start
  mlp_channel          w1's both_sides final (162 weights), channel move, then the group walk
  mlp_channel_sgd      the same with SGD (lr 0.1, momentum 0.9) for every retrain (variant; rotation-equivariant)
  mlp_channel_lr1e-2   the same with AdamW at lr 1e-2 (variant, for amplitude growth)

Commands (from experiments/, MODAL_PROFILE=chromatic, MODAL_BUILD_VALIDATION=warn):
    modal run grokking/whittle/whittle2.py::whittle2_gates
    modal run grokking/whittle/whittle2.py::whittle2_run --tag w2smoke --smoke 1
    modal run --detach grokking/whittle/whittle2.py::whittle2_run --tag w2
    python3 grokking/whittle/reduce_whittle2.py --tag w2 --fetch
"""

import json
import math
import os
import time

import numpy as np

from grokking.shared import DATA_DIR, LR, P, F, app, peak_rss_mb, torch, volume
from grokking.whittle.whittle import (BIL_BANK, F0, FLOOR, S_FEAT, Whittler, _pmap, build_arm, feats, load_final,
                                      structure, whittle_walk)

W1_TAG = "w1"
G_RETRAIN, G_SETTLE = 500, 100          # group moves: w1's budget
C_RETRAIN, C_SETTLE = 20000, 100        # channel moves and the plant: the long budget
MAX_ROUNDS = 600

ARMS2 = {
    "bil_channel": {"base": "bilinear_both", "start": "w1", "channel": True},
    "bil_channel_lr1e-2": {"base": "bilinear_both", "start": "w1", "channel": True, "lr": 1e-2},
    "bil_from_start": {"base": "bilinear_both", "start": "bank", "channel": True},
    "mlp_channel": {"base": "both_sides", "start": "w1", "channel": True},
    "mlp_channel_sgd": {"base": "both_sides", "start": "w1", "channel": True, "opt": "sgd", "lr": 0.1,
                        "momentum": 0.9},
    "mlp_channel_lr1e-2": {"base": "both_sides", "start": "w1", "channel": True, "lr": 1e-2},
}
PLANT_SYMS = (4, 9, 13)                 # the 3 symbols alive in w1's bilinear final (oracle DFT k 38, 18, 24)
for _t in PLANT_SYMS:
    ARMS2[f"bil_plant1_s{_t}"] = {"base": "bilinear_both", "start": "w1", "channel": False, "plant": _t}
# bil_channel_lr1e-2 is defined but not in the run of record: the bilinear's amplitude growth was not slow (NOTES w2)
MAIN_ARMS2 = ("bil_channel", "bil_plant1_s4", "bil_plant1_s9", "bil_plant1_s13", "bil_from_start", "mlp_channel",
              "mlp_channel_sgd", "mlp_channel_lr1e-2")


# =============================================================================
# Loading banked finals
# =============================================================================

def rebase(arm, Pm, split):
    """Recompute the arm's inputs from a banked Pm (the one the banked net was trained with)."""
    a_tr, b_tr, a_te, b_te = split
    a_all, b_all = np.repeat(np.arange(P), P), np.tile(np.arange(P), P)
    dd = arm["cfg"]["learner"] == "bilinear"
    arm["Pm"] = Pm
    arm["x_tr"] = feats(Pm, a_tr, b_tr, drop_dc=dd)
    arm["x_te"] = feats(Pm, a_te, b_te, drop_dc=dd)
    arm["x_all"] = feats(Pm, a_all, b_all, drop_dc=dd)


def load_banked_final(arm, path, split):
    z = np.load(path)
    rebase(arm, z["Pm"], split)
    sd = {k[6:]: torch.tensor(z[k]) for k in z.files if k.startswith("param/")}
    arm["model"].load_state_dict(sd)
    galive = {k[7:]: torch.tensor(z[k]).bool() for k in z.files if k.startswith("galive/")}
    bmask = {k[6:]: torch.tensor(z[k]) for k in z.files if k.startswith("bmask/")}
    return galive, bmask


# =============================================================================
# The w2 whittler: bias groups, channel moves, revival on fold, optimizer choice
# =============================================================================

class Whittler2(Whittler):
    def __init__(self, arm, y_own, split, wd=0.0, g_retrain=G_RETRAIN, g_settle=G_SETTLE, c_retrain=C_RETRAIN,
                 c_settle=C_SETTLE, opt="adamw", lr=LR, momentum=0.0, galive=None, bmask=None):
        super().__init__(arm, y_own, split, wd, g_retrain, g_settle)
        self.c_retrain, self.c_settle = int(c_retrain), int(c_settle)
        self.opt_name, self.lr, self.momentum = opt, float(lr), float(momentum)
        self.meta = arm["meta"]
        self.nsym = len(self.meta)
        colsym = torch.as_tensor(np.asarray(arm["colsym"]), dtype=torch.long)
        pm = _pmap(self.model)
        self.hb = "W" if self.kind == "bilinear" else "head"
        # bias groups: per unit (layer0 / layer1), per symbol (minted head), per class (plain head)
        self.B = {}
        if self.kind == "mlp":
            for b in ("layer0", "layer1"):
                n = pm[f"{b}.bias"].numel()
                self.B[b] = {"gid": torch.arange(n), "gsym": None}
        nb = pm[f"{self.hb}.bias"].numel()
        if self.model.out_map is not None:
            self.B[self.hb] = {"gid": colsym.clone(), "gsym": torch.arange(self.nsym)}
        else:
            self.B[self.hb] = {"gid": torch.arange(nb), "gsym": None}
        self.bmask[self.hb] = torch.ones(nb)
        for b, d in self.B.items():
            d["n_groups"] = int(d["gid"].max()) + 1
        # symbol of every group of every minted weight layer (-1: not minted)
        for n in self.order:
            L = self.L[n]
            g = torch.arange(L["n_groups"])
            if L["kind"] == "cols":
                cg = g % L["n_cg"]
                L["gsym"] = (cg % self.nsym) if self.kind == "mlp" else (cg + 1)
            elif L["kind"] == "rows":
                L["gsym"] = g // L["n_units"]
            else:
                L["gsym"] = torch.full((L["n_groups"],), -1)
        if galive is not None:
            for n, ga in galive.items():
                self.L[n]["galive"] = ga.clone()
        if bmask is not None:
            for b, m in bmask.items():
                self.bmask[b] = m.clone().float()
        self.apply_masks()
        self.border = [f"b:{b}" for b in self.B]
        self.stats.update({"n_revive": 0, "t_channel": 0.0})

    # --- bias groups ---------------------------------------------------------------------------
    def _bgalive(self, b):
        d = self.B[b]
        a = torch.zeros(d["n_groups"], dtype=torch.long)
        a.scatter_add_(0, d["gid"], (self.bmask[b] > 0).long())
        return a > 0

    def n_alive(self, lname):
        if lname.startswith("b:"):
            return int(self._bgalive(lname[2:]).sum())
        return super().n_alive(lname)

    def group_norms(self, lname):
        if lname.startswith("b:"):
            b = lname[2:]
            d = self.B[b]
            v = _pmap(self.model)[f"{b}.bias"].detach().double()
            s = torch.zeros(d["n_groups"], dtype=torch.float64)
            s.scatter_add_(0, d["gid"], v ** 2)
            return s.sqrt()
        return super().group_norms(lname)

    def prune(self, lname, n):
        if not lname.startswith("b:"):
            return super().prune(lname, n)
        b = lname[2:]
        nrm = self.group_norms(lname)
        ga = self._bgalive(b)
        idx = torch.nonzero(ga).flatten()
        order = idx[torch.argsort(nrm[idx], stable=True)]
        kill = torch.zeros(self.B[b]["n_groups"], dtype=torch.bool)
        kill[order[:n]] = True
        self.bmask[b][kill[self.B[b]["gid"]]] = 0
        self.apply_masks()

    # --- channels ------------------------------------------------------------------------------
    def channels_alive(self):
        out = set()
        for n in self.order:
            L = self.L[n]
            s = L["gsym"][L["galive"]]
            out |= set(int(x) for x in s[s >= 1])
        return sorted(out)

    def channel_norm(self, t):
        tot = 0.0
        for n in self.order:
            L = self.L[n]
            sel = (L["gsym"] == t) & L["galive"]
            if sel.any():
                tot += float((self.group_norms(n)[sel] ** 2).sum())
        d = self.B[self.hb]
        if d["gsym"] is not None:
            v = _pmap(self.model)[f"{self.hb}.bias"].detach().double()
            tot += float((v[(d["gid"] == t)] ** 2).sum())
        return math.sqrt(tot)

    def channels_by_norm(self):
        ch = self.channels_alive()
        nrm = {t: self.channel_norm(t) for t in ch}
        return sorted(ch, key=lambda t: (nrm[t], t)), nrm

    def remove_channel(self, t):
        for n in self.order:
            L = self.L[n]
            L["galive"] = L["galive"] & (L["gsym"] != t)
        d = self.B[self.hb]
        if d["gsym"] is not None:
            self.bmask[self.hb][d["gid"] == t] = 0
        self.apply_masks()

    def keep_only(self, t, drop_biases=True):
        """The plant: keep only symbol t's groups in every minted weight layer; zero every bias."""
        for n in self.order:
            L = self.L[n]
            L["galive"] = L["galive"] & (L["gsym"] == t)
        if drop_biases:
            for b in self.bmask:
                self.bmask[b].zero_()
        self.apply_masks()

    # --- cascade with revival --------------------------------------------------------------------
    @torch.no_grad()
    def cascade(self):
        if self.kind != "mlp":
            return super().cascade()
        t0 = time.time()
        pm = _pmap(self.model)
        hd = self.B[self.hb]
        while True:
            M0, M1, M2 = self.mask("layer0"), self.mask("layer1"), self.mask("head")
            in0, out0 = M0.any(1), M1.any(0)
            in1, out1 = M1.any(1), M2.any(0)
            b0m, b1m = self.bmask["layer0"].bool(), self.bmask["layer1"].bool()
            dead0 = ~out0 & (in0 | b0m)
            dead1 = ~out1 & (in1 | b1m)
            const0 = ~in0 & out0
            const1 = ~in1 & out1
            if not (dead0.any() or dead1.any() or const0.any() or const1.any()):
                break
            if const0.any():
                c = F.relu(pm["layer0.bias"][const0])
                delta = pm["layer1.weight"][:, const0] @ c
                pm["layer1.bias"].add_(delta)
                rv = (delta != 0) & ~self.bmask["layer1"].bool()
                self.bmask["layer1"][rv] = 1
                self.stats["n_revive"] += int(rv.sum())
                M1[:, const0] = 0
                self.bmask["layer0"][const0] = 0
                self.stats["n_fold0"] += int(const0.sum())
            if const1.any():
                c = F.relu(pm["layer1.bias"][const1])
                delta = pm["head.weight"][:, const1] @ c
                pm["head.bias"].add_(delta)
                touched = torch.zeros(hd["n_groups"], dtype=torch.long)
                touched.scatter_add_(0, hd["gid"], (delta != 0).long())
                ent = (touched > 0)[hd["gid"]] & ~self.bmask["head"].bool()
                self.bmask["head"][ent] = 1
                self.stats["n_revive"] += int(ent.sum())
                M2[:, const1] = 0
                self.bmask["layer1"][const1] = 0
                self.stats["n_fold1"] += int(const1.sum())
            M0[dead0, :] = 0
            self.bmask["layer0"][dead0] = 0
            M1[dead1, :] = 0
            self.bmask["layer1"][dead1] = 0
            for n, M in (("layer0", M0), ("layer1", M1), ("head", M2)):
                self._set_from_entry_mask(n, M)
            self.apply_masks()
        self.stats["t_book"] += time.time() - t0

    # --- retrain with a choice of optimizer --------------------------------------------------------
    def retrain(self, epochs=None, wd=None, settle=None, hook=None, hook_every=0):
        t0 = time.time()
        epochs = self.retrain_epochs if epochs is None else int(epochs)
        wd = self.wd if wd is None else float(wd)
        settle = self.settle_epochs if settle is None else int(settle)
        params = list(self.model.parameters())
        if self.opt_name == "sgd":
            opt = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=wd)
        pm = _pmap(self.model)
        gmask = {f"{n}.weight": self.mask(n) for n in self.order}
        gmask.update({f"{b}.bias": m for b, m in self.bmask.items()})
        pairs = [(pm[k], m) for k, m in gmask.items()]
        agree0, first_ok, streak, ran = None, None, 0, 0
        self.model.train()
        for e in range(epochs):
            opt.zero_grad()
            L = self.model(self.x_tr)
            with torch.no_grad():
                hit = L.argmax(1) == self.y_own
                ok = bool(hit.all())
                if e == 0:
                    agree0 = float(hit.float().mean())
                if ok and first_ok is None:
                    first_ok = e
                streak = streak + 1 if ok else 0
            if settle and streak >= settle:
                break
            F.cross_entropy(L, self.y_own).backward()
            ran = e + 1
            with torch.no_grad():
                for p_, m_ in pairs:
                    p_.grad.mul_(m_)
            opt.step()
            with torch.no_grad():
                for p_, m_ in pairs:
                    p_.mul_(m_)
            if hook is not None and hook_every and (e + 1) % hook_every == 0:
                self.model.eval()
                hook(e + 1)
                self.model.train()
        self.model.eval()
        self.stats["t_retrain"] += time.time() - t0
        return {"agree_post_prune": agree0, "first_ok_epoch": first_ok, "epochs": ran}

    def settle_channel(self):
        t0 = time.time()
        self.cascade()
        info = self.retrain(epochs=self.c_retrain, settle=self.c_settle)
        info.update(self.eval())
        self.stats["t_channel"] += time.time() - t0
        return info["n_disagree"] == 0, info

    # --- counts in both currencies ---------------------------------------------------------------
    def counts(self):
        out = {}
        ws = wg = 0
        for n in self.order:
            s, g = int(self.mask(n).sum()), self.n_alive(n)
            out[f"{n}_w"], out[f"{n}_g"] = s, g
            ws += s
            wg += g
        bs = bg = 0
        for b in self.B:
            s, g = int(self.bmask[b].sum()), int(self._bgalive(b).sum())
            out[f"b_{b}_s"], out[f"b_{b}_g"] = s, g
            bs += s
            bg += g
        out.update({"weights": ws, "weight_groups": wg, "bias_s": bs, "bias_g": bg,
                    "total_s": ws + bs, "total_g": wg + bg, "channels": self.channels_alive()})
        out.update(self.units())
        out["effective"] = ws + bs
        return out


# =============================================================================
# The w2 walk: channel phase <-> group phase
# =============================================================================

def whittle_walk2(pr, channel=True, f0=F0, floor=FLOOR, max_rounds=MAX_ROUNDS, on_admit=None, log=None,
                  ch_record=None):
    """Alternate a channel phase (to exhaustion) and a group phase (`whittle_walk` over weight + bias layers, to its
    floor) until a full cycle admits nothing. `pr` needs, beyond whittle_walk's interface, channels_by_norm /
    remove_channel / settle_channel. `ch_record()` returns before/after readouts for a channel attempt."""
    rounds = []
    n = 0
    ck = pr.save()
    cycle = 0
    layers = list(pr.order) + list(pr.border)
    while n < max_rounds:
        cycle += 1
        progressed = False
        if channel:
            while n < max_rounds:
                order, nrm = pr.channels_by_norm()
                if len(order) <= 1:
                    break
                before = ch_record() if ch_record is not None else {}
                admitted = False
                for t in order:
                    pr.remove_channel(t)
                    ok, info = pr.settle_channel()
                    n += 1
                    rec = {"round": n, "cycle": cycle, "move": "channel", "layer": f"ch{t}", "sym": int(t),
                           "channel_norm": nrm[t], "channels_before": list(order), "admit": bool(ok),
                           "n_prune": 1, "frac": None, "before": before}
                    rec.update(info)
                    if ok:
                        ck = pr.save()
                        if ch_record is not None:
                            rec["after"] = ch_record()
                        if on_admit is not None:
                            rec.update(on_admit())
                        admitted = True
                    else:
                        pr.restore(ck)
                    rounds.append(rec)
                    if log is not None:
                        log(rec)
                    if admitted or n >= max_rounds:
                        break
                if not admitted:
                    break
                progressed = True
        off = n

        def glog(r, off=off):
            r["round"] += off
            r["cycle"] = cycle
            r["move"] = "group"
            if log is not None:
                log(r)
        g = whittle_walk(pr, layers, f0=f0, floor=floor, max_rounds=max_rounds - n, on_admit=on_admit, log=glog)
        rounds += g["rounds"]
        n += g["n_rounds"]
        ck = pr.save()
        if any(r["admit"] for r in g["rounds"]):
            progressed = True
        if not channel or not progressed:
            break
    return {"rounds": rounds, "n_rounds": n, "cycles": cycle, "hit_cap": n >= max_rounds}


# =============================================================================
# Explicit read of a bilinear channel (Gauss's 3-multiplication form up to the rotation)
# =============================================================================

def _complex_basis():
    """E_re[j,k,l], E_im[j,k,l]: coefficients of y_a[j] y_b[k] y_c[l] in Re / Im of z_a z_b conj(z_c),
    z = y0 + i y1."""
    Ere, Eim = np.zeros((2, 2, 2)), np.zeros((2, 2, 2))
    for j in range(2):
        for k in range(2):
            for l in range(2):
                v = (1j ** j) * (1j ** k) * ((-1j) ** l)
                Ere[j, k, l], Eim[j, k, l] = v.real, v.imag
    return Ere, Eim


def explicit_bilinear(w, snap):
    """Per alive channel of a bilinear arm: its live units' U, V (on the symbol's 2 input columns) and W (on its 2
    head rows), the 2x2x2 tensor T = sum_i U_i (x) V_i (x) W_i, and the best fit T ~ A Re(e^{i psi} z_a z_b conj z_c)
    (the complex product up to a rotation of the frame). Also the symbol's eigenvalue phase (endogenous) and its DFT
    match (oracle), the frame phase at the identity symbol e_hat, and the margins on all p^2 pairs."""
    arm = w.arm
    colsym = np.asarray(arm["colsym"])
    m = w.model
    U, V = m.U.weight.detach().double().numpy(), m.V.weight.detach().double().numpy()
    Wh, bh = m.W.weight.detach().double().numpy(), m.W.bias.detach().double().numpy()
    MU, MV, MW = w.mask("U").numpy(), w.mask("V").numpy(), w.mask("W").numpy()
    live = (MU.any(1) & MV.any(1) & MW.any(0))
    Ere, Eim = _complex_basis()
    G = np.array([[np.sum(Ere * Ere), np.sum(Ere * Eim)], [np.sum(Eim * Ere), np.sum(Eim * Eim)]])
    Pm = arm["Pm"]
    out = {"channels": [], "dc_row_alive": bool(MW[colsym == 0].any()),
           "bias_alive": int((w.bmask[w.hb] > 0).sum()), "bias_norm": float(np.linalg.norm(bh))}
    for t in w.channels_alive():
        cin = np.where(colsym[1:] == t)[0]
        rout = np.where(colsym == t)[0]
        units = [int(i) for i in np.where(live)[0] if MU[i, cin].any() or MV[i, cin].any() or MW[rout, i].any()]
        T = np.einsum("ij,ik,li->jkl", U[np.ix_(units, cin)], V[np.ix_(units, cin)], Wh[np.ix_(rout, units)])
        rhs = np.array([np.sum(T * Ere), np.sum(T * Eim)])
        pq = np.linalg.solve(G, rhs)
        fit = pq[0] * Ere + pq[1] * Eim
        A = float(np.hypot(*pq))
        psi = float(np.arctan2(-pq[1], pq[0]))          # A Re(e^{i psi} z) = A cos psi Re z - A sin psi Im z
        y = S_FEAT * Pm[:, np.where(colsym == t)[0]]       # the symbol's coordinates, all residues
        eh = int(snap.e_hat)
        meta = w.meta[t]
        cand = meta["cand"]
        out["channels"].append({
            "sym": int(t), "k_dft": meta["k_dft"], "cand": cand,
            "lam_phase": float(snap.read["lam_phase"][cand]),
            "lam_phase_p_over_2pi": float(snap.read["lam_phase"][cand] * P / (2 * np.pi)),
            "units": units,
            "U": U[np.ix_(units, cin)].round(5).tolist(), "V": V[np.ix_(units, cin)].round(5).tolist(),
            "W": Wh[np.ix_(rout, units)].T.round(5).tolist(),
            "W_dc": Wh[colsym == 0][:, units].ravel().round(5).tolist(),
            "T": T.round(5).tolist(), "A": A, "psi": psi,
            "rel_resid": float(np.linalg.norm(T - fit) / (np.linalg.norm(T) + 1e-300)),
            "frame_phase_at_e_hat": float(np.arctan2(y[eh, 1], y[eh, 0])),
            "y_radius_min_max": [float(np.linalg.norm(y, axis=1).min()), float(np.linalg.norm(y, axis=1).max())]})
    with torch.no_grad():
        L = w.model(w.x_all).double().numpy()
    n = len(L)
    true = L[np.arange(n), w.y_all]
    Lm = L.copy()
    Lm[np.arange(n), w.y_all] = -np.inf
    marg = true - Lm.max(1)
    out["margins"] = {"min": float(marg.min()), "max": float(marg.max()), "mean": float(marg.mean()),
                      "q01_q50_q99": [float(q) for q in np.quantile(marg, [0.01, 0.5, 0.99])],
                      "n_nonpositive": int((marg <= 0).sum()), "n_pairs": int(n),
                      "min_train": float(marg[w.is_tr].min()), "min_heldout": float(marg[~w.is_tr].min())}
    out["_margins_all"] = marg
    return out


# =============================================================================
# One arm
# =============================================================================

def make_arm(name, loaded, data_dir, cfg=None):
    cfg = dict(ARMS2[name] if cfg is None else cfg)
    st, snap, K, split = loaded
    bil = None
    if cfg["base"] == "bilinear_both":
        z = np.load(os.path.join(data_dir, *BIL_BANK))
        bil = {k: z[k] for k in z.files}
    arm = build_arm(cfg["base"], st, snap, K, split, bil_state=bil)
    galive = bmask = None
    if cfg["start"] == "w1":
        galive, bmask = load_banked_final(arm, os.path.join(data_dir, W1_TAG, f"{cfg['base']}_final.npz"), split)
    return cfg, arm, galive, bmask


def run_arm2(name, data_dir=DATA_DIR, smoke=0, loaded=None, save_dir=None, verbose=True, cfg=None):
    t0 = time.time()
    loaded = loaded if loaded is not None else load_final(data_dir)
    st, snap, K, split = loaded
    cfg, arm, galive, bmask = make_arm(name, loaded, data_dir, cfg)
    c_ret = 300 if smoke else C_RETRAIN
    g_ret = 60 if smoke else G_RETRAIN
    w = Whittler2(arm, snap.own_tr, split, 0.0, g_ret, 20 if smoke else G_SETTLE, c_ret, 20 if smoke else C_SETTLE,
                  cfg.get("opt", "adamw"), cfg.get("lr", LR), cfg.get("momentum", 0.0), galive, bmask)
    rec = {"arm": name, "cfg": cfg, "meta": arm["meta"], "g_retrain": g_ret, "c_retrain": c_ret,
           "g_settle": w.settle_epochs, "c_settle": w.c_settle, "f0": F0, "floor": FLOOR, "smoke": smoke}
    rec["start"] = dict(w.eval(), **w.counts())
    rec["start"]["probe"] = w.probe()

    def probe_short():
        pr = w.probe()
        return {k: pr[k] for k in ("gbar_margin", "margin_tr_min", "margin_te_min", "wnorm")} | {"alpha": pr["alpha"]}

    def ch_record():
        order, nrm = w.channels_by_norm()
        return {"probe": probe_short(), "channel_norms": {str(t): nrm[t] for t in order}}

    if verbose:
        print(f"[{name}] start: {json.dumps({k: v for k, v in rec['start'].items() if k != 'probe'})}", flush=True)
    plant = None
    if cfg.get("plant") is not None:
        t = int(cfg["plant"])
        w.keep_only(t)
        w.cascade()
        c0 = w.counts()
        with torch.no_grad():
            pre = w.eval()
        info = w.retrain(epochs=c_ret, settle=w.c_settle)
        ev = w.eval()
        plant = {"sym": t, "k_dft": arm["meta"][t]["k_dft"], "counts_at_plant": c0, "eval_at_plant": pre,
                 "retrain": info, "eval_after": ev, "admit": ev["n_disagree"] == 0, "probe_after": probe_short()}
        rec["plant"] = plant
        if verbose:
            print(f"[{name}] plant s{t} (k {plant['k_dft']}): counts {c0['weights']} w / {c0.get('units')} units / "
                  f"{c0['bias_s']} bias; pre dis={pre['n_disagree']} -> after {info['epochs']} epochs dis="
                  f"{ev['n_disagree']} ho={ev['heldout']:.4f} gm={plant['probe_after']['gbar_margin']:.3f}", flush=True)
    rec["start_walk"] = dict(w.eval(), **w.counts())
    ck_start = w.save()

    def on_admit():
        return {"counts": w.counts(), "probe": {k: v for k, v in w.probe().items() if k != "alpha"}}

    def log(r):
        if not verbose:
            return
        c = r.get("counts", {})
        print(f"  [{name}] r{r['round']:3d} {r.get('move', 'group')[:2]} {r['layer']:>9s} -{r['n_prune']:4d} "
              f"{'ADMIT' if r['admit'] else 'reject'} dis={r['n_disagree']:4d} ho={r['heldout']:.4f} "
              f"ep={r['epochs']} ok@{r['first_ok_epoch']} w={c.get('weights', '-')} g={c.get('total_g', '-')} "
              f"ch={c.get('channels', '-')} t={time.time() - t0:.0f}s", flush=True)

    walk_ok = plant is None or plant["admit"]
    if walk_ok:
        walk = whittle_walk2(w, channel=cfg["channel"], max_rounds=12 if smoke else MAX_ROUNDS, on_admit=on_admit,
                             log=log, ch_record=ch_record)
    else:
        walk = {"rounds": [], "n_rounds": 0, "cycles": 0, "hit_cap": False, "skipped": "plant failed the gate"}
    rec["walk"] = walk
    rec["final"] = dict(w.eval(), **w.counts())
    rec["final"]["probe"] = w.probe()
    rec["final"]["logit_energy"] = w.logit_energy()
    rec["structure"] = structure(w, arm)
    if w.kind == "bilinear":
        ex = explicit_bilinear(w, snap)
        marg = ex.pop("_margins_all")
        rec["explicit"] = ex
    else:
        marg = None
    rec["stats"] = dict(w.stats)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        arrs = {f"param/{k}": v.detach().numpy() for k, v in w.model.state_dict().items()}
        arrs.update({f"galive/{n}": w.L[n]["galive"].numpy() for n in w.order})
        arrs.update({f"bmask/{b}": m.numpy() for b, m in w.bmask.items()})
        arrs["Pm"] = arm["Pm"]
        if marg is not None:
            arrs["margins_all_pairs"] = marg.astype(np.float32)
        np.savez_compressed(os.path.join(save_dir, f"{name}_final.npz"), **arrs)
    # drift twin: the walk's start state (after the plant, if any), nothing pruned, retrained with the admitted
    # rounds' epoch counts in order, each with the optimizer of the arm; held-out logged
    t1 = time.time()
    w.restore(ck_start)
    twin = []
    for x in walk["rounds"]:
        if x["admit"] and x["epochs"] > 0:
            w.retrain(epochs=x["epochs"], settle=0)
            ev = w.eval()
            twin.append({"round": x["round"], "epochs": x["epochs"], "heldout": ev["heldout"],
                         "n_disagree": ev["n_disagree"]})
    rec["drift_twin"] = twin
    rec["stats"]["t_twin"] = time.time() - t1
    rec["seconds"] = time.time() - t0
    rec["peak_rss_mb"] = peak_rss_mb()
    if verbose:
        f = rec["final"]
        print(f"[{name}] final: w={f['weights']} g={f['total_g']} (weight groups {f['weight_groups']}, bias groups "
              f"{f['bias_g']}) s={f['total_s']} ch={f['channels']} ho={f['heldout']:.4f} rounds={walk['n_rounds']} "
              f"{rec['seconds']:.0f}s stats={rec['stats']}", flush=True)
    return rec


# =============================================================================
# Gates
# =============================================================================

class MockPruner2:
    """G9: a designed walk. Weight layer A (n groups, gate passes iff pruned <= capA), channels {1, 2, 3}: removing
    channel c passes iff c in removable; channel norms order 3 < 1 < 2."""

    def __init__(self, n=40, capA=25, removable=(1,)):
        self.order, self.border = ["A"], []
        self.n, self.cap, self.removable = n, capA, set(removable)
        self.s = {"pruned": 0, "ch": {1, 2, 3}}

    def n_alive(self, L):
        return self.n - self.s["pruned"]

    def prune(self, L, k):
        self.s["pruned"] += k

    def settle(self):
        ok = self.s["pruned"] <= self.cap and not ({1, 2, 3} - self.s["ch"] - self.removable)
        return ok, {"n_disagree": 0 if ok else 1, "heldout": 1.0, "epochs": 1, "first_ok_epoch": 0,
                    "agree_post_prune": 1.0}

    def settle_channel(self):
        return self.settle()

    def channels_by_norm(self):
        order = [c for c in (3, 1, 2) if c in self.s["ch"]]
        return order, {c: float(c) for c in order}

    def remove_channel(self, t):
        self.s["ch"] = self.s["ch"] - {t}

    def save(self):
        return {"pruned": self.s["pruned"], "ch": set(self.s["ch"])}

    def restore(self, s):
        self.s = {"pruned": s["pruned"], "ch": set(s["ch"])}


def gates2(data_dir=DATA_DIR):
    t0 = time.time()
    loaded = load_final(data_dir)
    st, snap, K, split = loaded
    res = {}
    with open(os.path.join(data_dir, W1_TAG, "whittle.json")) as f:
        w1 = json.load(f)["results"]
    # G5: reload w1's banked finals
    for name, base in (("bil_channel", "bilinear_both"), ("mlp_channel", "both_sides")):
        cfg, arm, ga, bm = make_arm(name, loaded, data_dir)
        w = Whittler2(arm, snap.own_tr, split, galive=ga, bmask=bm)
        ev, c = w.eval(), w.counts()
        rf = w1[base]["final"]
        res[f"G5_{base}"] = {"weights": c["weights"], "w1_weights": rf["weights"], "n_disagree": ev["n_disagree"],
                             "heldout": ev["heldout"], "w1_heldout": rf["heldout"],
                             "pass": c["weights"] == rf["weights"] and ev["n_disagree"] == 0
                             and abs(ev["heldout"] - rf["heldout"]) < 1e-9,
                             "channels": c["channels"], "counts": c}
    # G6a: channel move on the bilinear: exactly the channel's units go; the function loses exactly their terms
    cfg, arm, ga, bm = make_arm("bil_channel", loaded, data_dir)
    w = Whittler2(arm, snap.own_tr, split, galive=ga, bmask=bm)
    colsym = np.asarray(arm["colsym"])
    ck = w.save()
    g6a = {}
    for t in w.channels_alive():
        w.restore(ck)
        m = w.model
        with torch.no_grad():
            L0 = m(w.x_tr).double()
            cin = torch.as_tensor(np.where(colsym[1:] == t)[0])
            rout = torch.as_tensor(np.where(colsym == t)[0])
            xa, xb = w.x_tr[:, :m.d_half], w.x_tr[:, m.d_half:]
            MU, MV, MW = w.mask("U"), w.mask("V"), w.mask("W")
            units_t = [i for i in range(MU.shape[0]) if MU[i, cin].any() and MU[i].sum() == MU[i, cin].sum()]
            ui = torch.as_tensor(units_t)
            h = (xa @ m.U.weight[ui].T) * (xb @ m.V.weight[ui].T)
            z = h @ m.W.weight[:, ui].T
            bt = torch.zeros_like(m.W.bias)
            bt[rout] = m.W.bias[rout]
            contrib = ((z + bt) @ m.out_map.T).double()
        w.remove_channel(t)
        w.cascade()
        c = w.counts()
        with torch.no_grad():
            L1 = w.model(w.x_tr).double()
        g6a[str(t)] = {"units_t": units_t, "units_after": c["units"], "units_before": 9,
                       "maxabs_vs_expected": float((L1 - (L0 - contrib)).abs().max()),
                       "weights_after": c["weights"], "channels_after": c["channels"]}
    res["G6a_bilinear_channel"] = g6a
    # G6b: channel move on the MLP: equals zeroing the channel's groups by hand (the cascade is exact)
    cfg, arm, ga, bm = make_arm("mlp_channel", loaded, data_dir)
    w = Whittler2(arm, snap.own_tr, split, galive=ga, bmask=bm)
    ck = w.save()
    g6b = {}
    for t in w.channels_alive():
        w.restore(ck)
        pm = _pmap(w.model)
        with torch.no_grad():
            saved = {k: v.clone() for k, v in pm.items()}
            L = w.L["layer0"]
            gsel = (L["gsym"] == t)[L["gid"]]
            pm["layer0.weight"][gsel] = 0
            Lh = w.L["head"]
            pm["head.weight"][(Lh["gsym"] == t)[Lh["gid"]]] = 0
            pm["head.bias"][w.B["head"]["gid"] == t] = 0
            ref = w.model(w.x_tr).double()
            for k, v in saved.items():
                pm[k].copy_(v)
        u0 = w.counts()
        w.remove_channel(t)
        w.cascade()
        with torch.no_grad():
            got = w.model(w.x_tr).double()
        c = w.counts()
        g6b[str(t)] = {"maxabs_vs_zeroed": float((got - ref).abs().max()), "units_before": [u0["units0"], u0["units1"]],
                       "units_after": [c["units0"], c["units1"]], "weights_after": c["weights"],
                       "revived": w.stats["n_revive"], "channels_after": c["channels"]}
    res["G6b_mlp_channel"] = g6b
    # G6c: fold with revival: prune the whole head bias, then silence all incoming weights of 3 layer-1 units
    w.restore(ck)
    w.prune("b:head", w.n_alive("b:head"))
    with torch.no_grad():
        M1 = w.mask("layer1")
        live1 = torch.nonzero(M1.any(1) & w.mask("head").any(0)).flatten()[:3]
        M1[live1, :] = 0
        w._set_from_entry_mask("layer1", M1)
        w.apply_masks()
        Lz = w.model(w.x_tr).clone()
        rv0 = w.stats["n_revive"]
        w.cascade()
        La = w.model(w.x_tr)
    res["G6c_revival"] = {"units_silenced": live1.tolist(), "maxabs": float((La - Lz).abs().max()),
                          "revived": w.stats["n_revive"] - rv0, "head_bias_alive_after": int(w.bmask["head"].sum())}
    # G7: each plant starts at exactly 3 units / 18 weights / 0 biases
    g7 = {}
    for t in PLANT_SYMS:
        cfg, arm, ga, bm = make_arm(f"bil_plant1_s{t}", loaded, data_dir)
        w = Whittler2(arm, snap.own_tr, split, galive=ga, bmask=bm)
        w.keep_only(t)
        w.cascade()
        c = w.counts()
        g7[str(t)] = {"k_dft": arm["meta"][t]["k_dft"], "units": c["units"], "weights": c["weights"],
                      "bias_s": c["bias_s"], "channels": c["channels"],
                      "pass": c["units"] == 3 and c["weights"] == 18 and c["bias_s"] == 0, "eval": w.eval()}
    res["G7_plant_start"] = g7
    # G8: zero-prune long retrain (no early stop) on w1's bilinear final (AdamW) and on w1's both_sides final
    # (AdamW and SGD): agreement must stay 1.0 on the bilinear; the MLP numbers decide the SGD variant
    g8 = {}
    for name, opt, lr, mom in (("bil_channel", "adamw", LR, 0.0), ("bil_channel", "adamw", 1e-2, 0.0),
                               ("mlp_channel", "adamw", LR, 0.0), ("mlp_channel", "sgd", 0.1, 0.9)):
        cfg, arm, ga, bm = make_arm(name, loaded, data_dir)
        w = Whittler2(arm, snap.own_tr, split, opt=opt, lr=lr, momentum=mom, galive=ga, bmask=bm)
        hist = []

        def hook(e, w=w, hist=hist):
            ev = w.eval()
            hist.append({"epochs": e, "n_disagree": ev["n_disagree"], "heldout": ev["heldout"],
                         "gbar_margin": w.probe()["gbar_margin"]})
        w.retrain(epochs=C_RETRAIN, settle=0, hook=hook, hook_every=C_RETRAIN // 8)
        g8[f"{name}_{opt}_lr{lr:g}"] = hist
    res["G8_long_retrain"] = g8
    res["G8_bilinear_pass"] = all(h["n_disagree"] == 0 for h in g8["bil_channel_adamw_lr0.001"])
    # G9: walk2 logic on a designed case
    m = MockPruner2()
    r = whittle_walk2(m, channel=True, max_rounds=500)
    tried = [(x["sym"], x["admit"]) for x in r["rounds"] if x.get("move") == "channel"]
    res["G9_walk2"] = {"channels_left": sorted(m.s["ch"]), "pruned": m.s["pruned"], "channel_tries": tried,
                       "pass": sorted(m.s["ch"]) == [2, 3] and m.s["pruned"] == 25
                       and tried[:4] == [(3, False), (1, True), (3, False), (2, False)]}
    res["seconds"] = time.time() - t0
    res["peak_rss_mb"] = peak_rss_mb()
    return res


# =============================================================================
# Modal entrypoints (prefixed `whittle2_`)
# =============================================================================

@app.function(cpu=4.0, memory=2048, timeout=3600, volumes={DATA_DIR: volume})
def whittle2_gates():
    volume.reload()
    res = gates2(DATA_DIR)
    for k, v in res.items():
        print(f"{k}: {v}", flush=True)
    d = os.path.join(DATA_DIR, "whittle_gates")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "gates2.json"), "w") as f:
        json.dump(res, f)
    volume.commit()
    return {k: res[k] for k in ("G8_bilinear_pass", "seconds")}


@app.function(cpu=4.0, memory=1536, timeout=4 * 3600, volumes={DATA_DIR: volume}, max_containers=8)
def whittle2_arm(tag: str, name: str, smoke: int):
    volume.reload()
    d = os.path.join(DATA_DIR, tag)
    rec = run_arm2(name, DATA_DIR, smoke=smoke, save_dir=d)
    with open(os.path.join(d, f"{name}.json"), "w") as f:
        json.dump(rec, f)
    volume.commit()
    return rec


@app.function(cpu=1.0, memory=1024, timeout=5 * 3600, volumes={DATA_DIR: volume})
def whittle2_run(tag: str = "w2", arms: str = ",".join(MAIN_ARMS2), smoke: int = 0):
    t0 = time.time()
    names = [a for a in arms.split(",") if a]
    print(f"{len(names)} arms {names} smoke={smoke}", flush=True)
    recs = list(whittle2_arm.starmap([(tag, n, smoke) for n in names]))
    out = {"tag": tag, "arms": names, "smoke": smoke, "g_retrain": G_RETRAIN, "g_settle": G_SETTLE,
           "c_retrain": C_RETRAIN, "c_settle": C_SETTLE, "results": {r["arm"]: r for r in recs},
           "seconds": time.time() - t0}
    d = os.path.join(DATA_DIR, tag)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "whittle2.json")
    with open(path, "w") as f:
        json.dump(out, f)
    volume.commit()
    print(f"whittle2 done in {time.time() - t0:.0f}s -> {path}; per arm: "
          f"{[(r['arm'], r['final']['weights'], r['final']['total_g'], r['final']['channels'], round(r['final']['heldout'], 4), round(r['seconds'])) for r in recs]}",
          flush=True)
    return path
