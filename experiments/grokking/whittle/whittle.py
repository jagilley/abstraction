"""[grokking/whittle] Whittle: the grokked net continues under a new objective, "keep your modular
addition, and zero out as many weights as you can", with its first layer rewritten in the coordinates
Basis read off its own behaviour.

No fresh training. Every arm starts from m1's banked final net (/data/m1/true/snapshots.npz, label e39999,
seed 42) or, for the optional secondary, from r2's banked minted-13 bilinear net.

THE CHANGE OF VARIABLES. `BasisSnapshot` (basis/basis.py) diagonalizes the net's own operator
M_g[c, a] = softmax(net(a, g))[c] and returns 48 real 2-D subspaces Q_j plus a DC vector u0. The kept 13
are the census walk's candidates under form C / rule lt1 (reproduced here with `_walk_K`). The minted
coordinates are Pm = [u0 | Q_j1 | ... | Q_j13] orthonormalized by QR in that order (97 x 27; the columns
stay grouped by symbol, see NOTES decision 2). Input symbol a is spelled s * Pm[a, :] with s = sqrt(p/2)
(unit-amplitude waves, as in rung.py's char_features), and layer 0 becomes A = [Wa Pm, Wb Pm] / s, so
A x equals form C's layer 0 exactly. A stays trainable: the subspace is committed, the rows are not.

THE LOOP (`whittle_walk`). Per-layer round robin. A round prunes the ceil(f_L * n_alive) smallest surviving
groups of one layer L (group L2 norm; a group is one scalar, or in a minted layer one unit's connection to
one minted symbol: a 2-vector for a pair, a scalar for DC), removes weights that can no longer reach the
output (the exact cascade, `Whittler.cascade`), retrains full-batch (fresh AdamW, lr 1e-3, the arm's
weight decay, pruned entries held at exactly zero; the retrain stops once train agreement has held for
`settle` = 100 consecutive epochs, at most `retrain` = 2000 epochs), and is admitted iff the net's argmax
on the 2,822 train pairs agrees 100% with the original grokked net's train argmax (banked at the start).
On failure the last admitted state is restored and f_L is halved. The walk stops when every f_L < floor.
Held-out accuracy and the diagonal-margin probe are logged at every admitted round and never consumed.

ARMS (all MLP arms from the same banked net, seed 42):
    committed      layer 0 in the 13 minted coordinates (+ DC); layers 1-2 as grokked; wd 0
    unrestricted   the grokked net in its one-hot coordinates; wd 0
    both_sides     committed, and the head in minted coordinates too: logits = s Pm (B h + beta),
                   B = Pm^T W2 / s, beta = Pm^T b2 / s; wd 0
    all48          layer 0 in the full recovered basis [u0 | Q_1 .. Q_48] (97 x 97, a rotation); wd 0
    committed_wd1  committed with weight decay 1.0 during retraining
    bilinear_both  (optional secondary) r2's banked minted-13 bilinear product net (width 52): input
                   rotated from rung's DFT cos/sin features into the recovered 13 pairs, head minted; wd 0

Commands (run from experiments/, MODAL_PROFILE=chromatic, MODAL_BUILD_VALIDATION=warn):
    modal run grokking/whittle/whittle.py::whittle_gates
    modal run grokking/whittle/whittle.py::whittle_run --tag wsmoke --smoke 1
    modal run --detach grokking/whittle/whittle.py::whittle_run --tag w1
    python3 grokking/whittle/reduce_whittle.py --tag w1 --fetch
"""

import json
import math
import os
import time

import numpy as np

from grokking.shared import (DATA_DIR, HIDDEN, LR, P, F, app, char_features, nn, onehot, peak_rss_mb,
                             split_pairs, torch, volume)
from grokking.mint import _load_snaps, gate_rules, logit_table_energy
from grokking.basis.basis import BasisSnapshot, _walk_K, overlap_matrix
from grokking.rung_controls import K13, make_probe

MINT_TAG = "m1"
FINAL = "e39999"
BIL_BANK = ("r2", "B3_minted13_bil_stop_final.npz")
S_FEAT = math.sqrt(P / 2.0)       # unit-amplitude waves: s * Q_j[a, :] = (cos, sin) of the symbol's angle
F0 = 0.5                          # initial prune fraction per layer (of that layer's surviving groups)
FLOOR = 0.005                     # a layer stops once its fraction is halved below this (7 halvings)
RETRAIN = 500                     # max epochs per round (G3 is checked at this full budget, no early stop;
                                  # all48 fails G3 at 1000 and 2000: NOTES decision 6)
SETTLE = 100                      # a round's retrain stops once train agreement has held for this many epochs
MAX_ROUNDS = 400                  # safety cap per arm
RECOVER_CHUNKS = 20               # both_sides / bilinear: retrain up to this many budgets to recover agreement

ARMS = {
    "committed": {"learner": "mlp", "basis": "K", "head": "plain", "wd": 0.0},
    "unrestricted": {"learner": "mlp", "basis": "onehot", "head": "plain", "wd": 0.0},
    "both_sides": {"learner": "mlp", "basis": "K", "head": "minted", "wd": 0.0},
    "all48": {"learner": "mlp", "basis": "all", "head": "plain", "wd": 0.0},
    "committed_wd1": {"learner": "mlp", "basis": "K", "head": "plain", "wd": 1.0},
    "bilinear_both": {"learner": "bilinear", "basis": "K", "head": "minted", "wd": 0.0},
    # budget sensitivity (added; NOTES decision 6): the main arm with a 4x retrain cap (G3 passes for it at 2000)
    "committed_R2000": {"learner": "mlp", "basis": "K", "head": "plain", "wd": 0.0, "retrain": 2000},
}
MAIN_ARMS = ("committed", "unrestricted", "both_sides", "all48", "committed_wd1")


# =============================================================================
# The banked net and its minted coordinates
# =============================================================================

def load_final(data_dir=DATA_DIR, mint_tag=MINT_TAG):
    """m1's final net, its BasisSnapshot, and the kept candidates K (form C, rule lt1)."""
    run = os.path.join(data_dir, mint_tag, "true")
    labels, _, states = _load_snaps(os.path.join(run, "snapshots.npz"))
    st = states[labels.index(FINAL)]
    tl = np.load(os.path.join(run, "train_labels.npy"))
    split = split_pairs()
    snap = BasisSnapshot(st, split, train_labels=tl)
    K = tuple(int(j) for j in _walk_K(snap, "C", gate_rules()["lt1"]))
    return st, snap, K, split


def minted_basis(snap, cand):
    """Pm (p x d): [u0 | Q_c for c in cand] orthonormalized by QR in that order. Same span as
    basis.orth(...) (so the projector is form C's); unlike an SVD orth, QR keeps each symbol's columns inside
    its own subspace up to the subspaces' mutual non-orthogonality. colsym[i] = symbol of column i
    (0 = DC, t = cand[t-1])."""
    blocks = [snap.u0[:, None]] + [snap.Qs[j] for j in cand]
    B = np.concatenate(blocks, 1)
    Q, R = np.linalg.qr(B)
    Q = Q * np.sign(np.diag(R))
    colsym = np.concatenate([np.full(b.shape[1], t) for t, b in enumerate(blocks)])
    G = B.T @ B
    ov = [float(overlap_matrix([Q[:, colsym == t]], [blocks[t]])[0, 0]) for t in range(len(blocks))]
    info = {"d": int(Q.shape[1]), "gram_offdiag_max": float(np.abs(G - np.eye(len(G))).max()),
            "qr_col_change_max": float(np.abs(Q - B).max()), "sym_overlap_min": float(min(ov)),
            "orth_err": float(np.abs(Q.T @ Q - np.eye(Q.shape[1])).max())}
    return Q, colsym, info


def sym_meta(snap, cand):
    """Per minted symbol: candidate index, producer rank, and the oracle's DFT match (logged only)."""
    rd = snap.read
    rank = {int(j): r for r, j in enumerate(snap.producer_order())}
    out = [{"sym": 0, "name": "DC", "cand": None, "dim": 1, "prod_rank": None, "k_dft": 0, "dft_ov": None}]
    for t, j in enumerate(cand):
        out.append({"sym": t + 1, "name": f"s{t + 1}", "cand": int(j), "dim": int(snap.Qs[j].shape[1]),
                    "prod_rank": rank[int(j)], "k_dft": int(rd["k_match"][j]), "dft_ov": float(rd["dft_ov"][j])})
    return out


def feats(Pm, a, b, s=S_FEAT, drop_dc=False):
    Pu = Pm[:, 1:] if drop_dc else Pm
    return torch.tensor(np.concatenate([s * Pu[a], s * Pu[b]], 1), dtype=torch.float32)


# =============================================================================
# The nets
# =============================================================================

if nn is not None:

    class WNet(nn.Module):
        """GrokMLP with a free input dim and an optional minted head: logits = head(h2) @ out_map^T."""

        def __init__(self, d_in, d_head, out_map=None, hidden=HIDDEN):
            super().__init__()
            self.layer0 = nn.Linear(d_in, hidden[0])
            self.layer1 = nn.Linear(hidden[0], hidden[1])
            self.head = nn.Linear(hidden[1], d_head)
            self.register_buffer("out_map", out_map)

        def forward(self, x):
            z = self.head(F.relu(self.layer1(F.relu(self.layer0(x)))))
            return z if self.out_map is None else z @ self.out_map.T

    class WBil(nn.Module):
        """rung.py's Bilinear, logits = W((U x_a) * (V x_b)) + b, with an optional minted head."""

        def __init__(self, d_half, m, d_head, out_map=None):
            super().__init__()
            self.d_half = d_half
            self.U = nn.Linear(d_half, m, bias=False)
            self.V = nn.Linear(d_half, m, bias=False)
            self.W = nn.Linear(m, d_head)
            self.register_buffer("out_map", out_map)

        def forward(self, x):
            z = self.W(self.U(x[:, :self.d_half]) * self.V(x[:, self.d_half:]))
            return z if self.out_map is None else z @ self.out_map.T


def _gid_scalar(shape):
    return torch.arange(int(np.prod(shape))).view(*shape)


def _gid_cols(n_rows, colgroup):
    """Entry (i, c) -> i * n_cg + colgroup[c]: groups of one row (one unit's connection to one symbol)."""
    cg = torch.as_tensor(np.asarray(colgroup), dtype=torch.long)
    n_cg = int(cg.max()) + 1
    return torch.arange(n_rows)[:, None] * n_cg + cg[None, :], n_cg


def _gid_rows(rowgroup, n_cols):
    """Entry (r, j) -> rowgroup[r] * n_cols + j: groups of one column (one unit's write to one symbol)."""
    rg = torch.as_tensor(np.asarray(rowgroup), dtype=torch.long)
    return rg[:, None] * n_cols + torch.arange(n_cols)[None, :], int(rg.max()) + 1


def build_arm(name, st, snap, K, split, bil_state=None, cfg=None):
    """The arm's net at initialization, its inputs, its prunable layers and its symbol metadata."""
    cfg = dict(ARMS[name] if cfg is None else cfg)
    a_tr, b_tr, a_te, b_te = split
    a_all, b_all = np.repeat(np.arange(P), P), np.tile(np.arange(P), P)
    prod = [int(j) for j in snap.producer_order()]
    Kset = set(K)
    cand = [j for j in prod if j in Kset] if cfg["basis"] == "K" else prod     # onehot: read in the full basis
    Pm, colsym, binfo = minted_basis(snap, cand)
    meta = sym_meta(snap, cand)
    nsym = len(meta)
    arm = {"name": name, "cfg": cfg, "basis_info": binfo, "meta": meta, "cand": cand, "Pm": Pm,
           "colsym": colsym}
    if cfg["learner"] == "mlp":
        W0 = np.asarray(st["layer0.weight"], np.float64)
        Wa, Wb = W0[:, :P], W0[:, P:]
        if cfg["basis"] == "onehot":
            fx = onehot
            A = W0
            l0 = {"gid": _gid_scalar(A.shape), "kind": "scalar"}
        else:
            def fx(a, b, Pm=Pm):
                return feats(Pm, a, b)
            A = np.concatenate([Wa @ Pm, Wb @ Pm], 1) / S_FEAT
            gid, n_cg = _gid_cols(A.shape[0], np.concatenate([colsym, colsym + nsym]))
            l0 = {"gid": gid, "kind": "cols", "n_cg": n_cg}          # cg < nsym: operand a; >= nsym: b
        W2 = np.asarray(st["head.weight"], np.float64)
        b2 = np.asarray(st["head.bias"], np.float64)
        if cfg["head"] == "minted":
            Bh, beta = Pm.T @ W2 / S_FEAT, Pm.T @ b2 / S_FEAT
            out_map = torch.tensor(S_FEAT * Pm, dtype=torch.float32)
            gid, _ = _gid_rows(colsym, Bh.shape[1])
            l2 = {"gid": gid, "kind": "rows", "n_units": Bh.shape[1]}
        else:
            Bh, beta, out_map = W2, b2, None
            l2 = {"gid": _gid_scalar(W2.shape), "kind": "scalar"}
        model = WNet(A.shape[1], Bh.shape[0], out_map)
        sd = {"layer0.weight": A, "layer0.bias": st["layer0.bias"], "layer1.weight": st["layer1.weight"],
              "layer1.bias": st["layer1.bias"], "head.weight": Bh, "head.bias": beta}
        if out_map is not None:
            sd["out_map"] = out_map
        model.load_state_dict({k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in sd.items()})
        layers = [("layer0", l0), ("layer1", {"gid": _gid_scalar(model.layer1.weight.shape), "kind": "scalar"}),
                  ("head", l2)]
        arm["bias_masked"] = ["layer0", "layer1"]
    else:
        # r2's banked minted-13 bilinear: rung's input is DFT cos/sin at K13 (sorted); rotate it into the
        # recovered pairs (no DC on the input, as in rung): x_dft = x_rec @ T, so U' = U T^T.
        U, V = np.asarray(bil_state["U.weight"], np.float64), np.asarray(bil_state["V.weight"], np.float64)
        Wh, bh = np.asarray(bil_state["W.weight"], np.float64), np.asarray(bil_state["W.bias"], np.float64)
        ar = np.arange(P)
        X_dft = char_features(ar, ar, K13).double().numpy()[:, :2 * len(K13)]
        X_rec = S_FEAT * Pm[:, 1:]
        T = np.linalg.lstsq(X_rec, X_dft, rcond=None)[0]
        arm["basis_info"]["bil_rot_resid"] = float(np.abs(X_rec @ T - X_dft).max())
        arm["basis_info"]["bil_rot_orth_err"] = float(np.abs(T.T @ T - np.eye(T.shape[0])).max())
        U2, V2 = U @ T.T, V @ T.T

        def fx(a, b, Pm=Pm):
            return feats(Pm, a, b, drop_dc=True)
        Bh, beta = Pm.T @ Wh / S_FEAT, Pm.T @ bh / S_FEAT
        out_map = torch.tensor(S_FEAT * Pm, dtype=torch.float32)
        model = WBil(U2.shape[1], U2.shape[0], Bh.shape[0], out_map)
        model.load_state_dict({"U.weight": torch.tensor(U2, dtype=torch.float32),
                               "V.weight": torch.tensor(V2, dtype=torch.float32),
                               "W.weight": torch.tensor(Bh, dtype=torch.float32),
                               "W.bias": torch.tensor(beta, dtype=torch.float32), "out_map": out_map})
        cgin = colsym[1:] - 1
        gU, n_cg = _gid_cols(U2.shape[0], cgin)
        gW, _ = _gid_rows(colsym, Bh.shape[1])
        layers = [("U", {"gid": gU, "kind": "cols", "n_cg": n_cg}),
                  ("V", {"gid": gU.clone(), "kind": "cols", "n_cg": n_cg}),
                  ("W", {"gid": gW, "kind": "rows", "n_units": Bh.shape[1]})]
        arm["bias_masked"] = []
        arm["bil_orig"] = {"U": U, "V": V, "W": Wh, "b": bh}
        # the bilinear's own original function, for G1
        def orig_fn(a, b):
            xd = char_features(a, b, K13).double().numpy()
            h = (xd[:, :26] @ U.T) * (xd[:, 26:] @ V.T)
            return h @ Wh.T + bh
        arm["orig_fn"] = orig_fn
    arm.update({"model": model, "layers": layers, "fx": fx,
                "x_tr": fx(a_tr, b_tr), "x_te": fx(a_te, b_te), "x_all": fx(a_all, b_all),
                "y_te": torch.as_tensor((a_te + b_te) % P)})
    return arm


# =============================================================================
# The whittler: masks, the exact cascade, retraining, the gate, readouts
# =============================================================================

def _pmap(model):
    return dict(model.named_parameters())


class Whittler:
    """Holds one arm's net, its group masks and bias masks, and runs prune / cascade / retrain / gate."""

    def __init__(self, arm, y_own, split, wd, retrain, settle=SETTLE, probe_freqs=K13):
        self.arm = arm
        self.model = arm["model"]
        self.kind = arm["cfg"]["learner"]
        self.wd = float(wd)
        self.retrain_epochs = int(retrain)
        self.settle_epochs = int(settle)
        self.x_tr, self.x_te, self.x_all = arm["x_tr"], arm["x_te"], arm["x_all"]
        self.y_own = torch.as_tensor(np.asarray(y_own))
        self.y_te = arm["y_te"]
        a_tr, b_tr, _, _ = split
        a_all, b_all = np.repeat(np.arange(P), P), np.tile(np.arange(P), P)
        tr = set(zip(a_tr.tolist(), b_tr.tolist()))
        self.is_tr = np.array([(int(x), int(y)) in tr for x, y in zip(a_all, b_all)])
        self.y_all = (a_all + b_all) % P
        self.probe_fn = make_probe(self.x_all, self.y_all, self.is_tr, list(probe_freqs))
        pm = _pmap(self.model)
        self.L = {}
        for lname, spec in arm["layers"]:
            w = pm[f"{lname}.weight"]
            gid = spec["gid"].reshape(w.shape)
            n_g = int(gid.max()) + 1
            self.L[lname] = dict(spec, w=w, gid=gid, n_groups=n_g,
                                 gsize=torch.bincount(gid.flatten(), minlength=n_g),
                                 galive=torch.ones(n_g, dtype=torch.bool))
        self.order = [n for n, _ in arm["layers"]]
        self.bmask = {b: torch.ones_like(pm[f"{b}.bias"]) for b in arm["bias_masked"]}
        self.stats = {"n_fold0": 0, "n_fold1": 0, "t_retrain": 0.0, "t_probe": 0.0, "t_book": 0.0}

    # --- masks -------------------------------------------------------------------------------
    def mask(self, lname):
        L = self.L[lname]
        return L["galive"][L["gid"]].to(L["w"].dtype)

    def _set_from_entry_mask(self, lname, M):
        L = self.L[lname]
        alive = torch.zeros(L["n_groups"], dtype=torch.long)
        alive.scatter_add_(0, L["gid"].flatten(), (M.flatten() > 0).long())
        L["galive"] = alive > 0

    @torch.no_grad()
    def apply_masks(self):
        pm = _pmap(self.model)
        for n in self.order:
            self.L[n]["w"].mul_(self.mask(n))
        for b, m in self.bmask.items():
            pm[f"{b}.bias"].mul_(m)

    def n_alive(self, lname):
        return int(self.L[lname]["galive"].sum())

    def group_norms(self, lname):
        L = self.L[lname]
        s = torch.zeros(L["n_groups"], dtype=torch.float64)
        s.scatter_add_(0, L["gid"].flatten(), L["w"].detach().double().flatten() ** 2)
        return s.sqrt()

    def prune(self, lname, n):
        """Zero the n smallest-norm surviving groups of one layer (ties by group index)."""
        L = self.L[lname]
        nrm = self.group_norms(lname)
        idx = torch.nonzero(L["galive"]).flatten()
        order = idx[torch.argsort(nrm[idx], stable=True)]
        L["galive"][order[:n]] = False
        self.apply_masks()

    # --- the exact cascade -------------------------------------------------------------------
    @torch.no_grad()
    def cascade(self):
        """Remove every weight that can no longer reach the output, and fold constant units into the next
        bias. Function-preserving (G2b). MLP: a layer-0 / layer-1 unit with no surviving outgoing weight loses
        its incoming weights and bias; a unit with no surviving incoming weight outputs relu(bias), which is
        added to the next layer's bias, and its outgoing weights are removed. Bilinear: a product unit with
        U-row, V-row or W-column empty is removed whole."""
        t0 = time.time()
        pm = _pmap(self.model)
        if self.kind == "mlp":
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
                    pm["layer1.bias"].add_(pm["layer1.weight"][:, const0] @ c)
                    M1[:, const0] = 0
                    self.bmask["layer0"][const0] = 0
                    self.stats["n_fold0"] += int(const0.sum())
                if const1.any():
                    c = F.relu(pm["layer1.bias"][const1])
                    pm["head.bias"].add_(pm["head.weight"][:, const1] @ c)
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
        else:
            while True:
                MU, MV, MW = self.mask("U"), self.mask("V"), self.mask("W")
                live = MU.any(1) & MV.any(1) & MW.any(0)
                touched = MU.any(1) | MV.any(1) | MW.any(0)
                dead = ~live & touched
                if not dead.any():
                    break
                MU[dead, :] = 0
                MV[dead, :] = 0
                MW[:, dead] = 0
                for n, M in (("U", MU), ("V", MV), ("W", MW)):
                    self._set_from_entry_mask(n, M)
                self.apply_masks()
        self.stats["t_book"] += time.time() - t0

    # --- retrain and gate --------------------------------------------------------------------
    @torch.no_grad()
    def eval(self):
        L_tr = self.model(self.x_tr)
        L_te = self.model(self.x_te)
        return {"train_agree": float((L_tr.argmax(1) == self.y_own).float().mean()),
                "n_disagree": int((L_tr.argmax(1) != self.y_own).sum()),
                "heldout": float((L_te.argmax(1) == self.y_te).float().mean()),
                "loss": float(F.cross_entropy(L_tr, self.y_own))}

    def retrain(self, epochs=None, wd=None, settle=None):
        """Full-batch AdamW (fresh state), lr 1e-3, on the train pairs with the original net's train argmax as
        targets (= the train labels here). Gradients of pruned entries are zeroed and every step is followed
        by re-masking, so pruned weights stay exactly zero. Stops early, before the step, once the pre-step
        forward has agreed on every train pair for `settle` consecutive epochs (0: never stop early); at most
        `epochs`. Returns the agreement right after the prune (epoch 0, pre-step), the first epoch at which it
        is 1.0, and the epochs run."""
        t0 = time.time()
        epochs = self.retrain_epochs if epochs is None else int(epochs)
        wd = self.wd if wd is None else float(wd)
        settle = self.settle_epochs if settle is None else int(settle)
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=wd)
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
                ok = bool((L.argmax(1) == self.y_own).all())
                if e == 0:
                    agree0 = float((L.argmax(1) == self.y_own).float().mean())
                if ok and first_ok is None:
                    first_ok = e
                streak = streak + 1 if ok else 0
            if settle and streak >= settle:
                break
            loss = F.cross_entropy(L, self.y_own)
            loss.backward()
            ran = e + 1
            with torch.no_grad():
                for p_, m_ in pairs:
                    p_.grad.mul_(m_)
            opt.step()
            with torch.no_grad():
                for p_, m_ in pairs:
                    p_.mul_(m_)
        self.model.eval()
        self.stats["t_retrain"] += time.time() - t0
        return {"agree_post_prune": agree0, "first_ok_epoch": first_ok, "epochs": ran}

    def settle(self):
        """cascade -> retrain -> gate. Returns (admitted, info)."""
        self.cascade()
        info = self.retrain()
        info.update(self.eval())
        return info["n_disagree"] == 0, info

    # --- save / restore ----------------------------------------------------------------------
    def save(self):
        t0 = time.time()
        s = {"sd": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
             "galive": {n: self.L[n]["galive"].clone() for n in self.order},
             "bmask": {b: m.clone() for b, m in self.bmask.items()}}
        self.stats["t_book"] += time.time() - t0
        return s

    def restore(self, s):
        t0 = time.time()
        self.model.load_state_dict(s["sd"])
        for n in self.order:
            self.L[n]["galive"] = s["galive"][n].clone()
        self.bmask = {b: m.clone() for b, m in s["bmask"].items()}
        self.stats["t_book"] += time.time() - t0

    # --- readouts ----------------------------------------------------------------------------
    def units(self):
        if self.kind == "mlp":
            M0, M1, M2 = self.mask("layer0"), self.mask("layer1"), self.mask("head")
            return {"units0": int((M0.any(1) & M1.any(0)).sum()), "units1": int((M1.any(1) & M2.any(0)).sum()),
                    "bias0_live": int(self.bmask["layer0"].sum()), "bias1_live": int(self.bmask["layer1"].sum())}
        MU, MV, MW = self.mask("U"), self.mask("V"), self.mask("W")
        return {"units": int((MU.any(1) & MV.any(1) & MW.any(0)).sum())}

    def counts(self):
        out = {}
        tot = 0
        for n in self.order:
            s = int(self.mask(n).sum())
            out[f"{n}_w"] = s
            out[f"{n}_g"] = self.n_alive(n)
            tot += s
        out["weights"] = tot
        u = self.units()
        out.update(u)
        pm = _pmap(self.model)
        head_bias = pm["head.bias" if self.kind == "mlp" else "W.bias"].numel()
        out["head_bias"] = int(head_bias)
        out["effective"] = tot + head_bias + (u["bias0_live"] + u["bias1_live"] if self.kind == "mlp" else 0)
        return out

    def probe(self):
        t0 = time.time()
        pr = self.probe_fn(self.model)
        keep = ("gbar_margin", "gbar0", "margin_te_min", "margin_te_mean", "margin_tr_min", "margin_tr_mean",
                "diag_frac", "wnorm", "alpha_sum_freqs")
        out = {k: pr[k] for k in keep}
        out["alpha"] = pr["alpha"]
        self.stats["t_probe"] += time.time() - t0
        return out

    @torch.no_grad()
    def logit_energy(self):
        L_all = self.model(self.x_all).double().numpy().reshape(P, P, P)
        lt = logit_table_energy(L_all)
        diag = np.asarray(lt["diag"])
        order = np.argsort(-diag)
        c = np.cumsum(diag[order]) / (diag.sum() + 1e-300)
        return {"diag_total": lt["diag_total"], "dc_frac": lt["dc_frac"],
                "diag": [round(float(x), 6) for x in diag],
                "top_k": [int(k) + 1 for k in order[:15]],
                "n90": int(np.searchsorted(c, 0.90) + 1), "n99": int(np.searchsorted(c, 0.99) + 1)}


# =============================================================================
# The walk (the census gate run in reverse over the net's own weights)
# =============================================================================

def whittle_walk(pr, layers, f0=F0, floor=FLOOR, max_rounds=MAX_ROUNDS, on_admit=None, log=None):
    """Per-layer round robin. For each layer L with f_L >= floor: prune ceil(f_L * alive_L) groups, settle
    (cascade, retrain, gate), admit iff the gate passes; on failure restore the last admitted state and
    halve f_L. Stops when every f_L < floor (or at max_rounds). `pr` needs n_alive / prune / settle / save /
    restore; `on_admit()` returns extra readouts for an admitted round."""
    frac = {L: float(f0) for L in layers}
    ck = pr.save()
    rounds = []
    n = 0
    while n < max_rounds and any(frac[L] >= floor for L in layers):
        for L in layers:
            if frac[L] < floor or n >= max_rounds:
                continue
            alive = pr.n_alive(L)
            if alive == 0:
                frac[L] = 0.0
                continue
            k = min(alive, max(1, math.ceil(frac[L] * alive - 1e-9)))
            pr.prune(L, k)
            ok, info = pr.settle()
            n += 1
            rec = {"round": n, "layer": L, "frac": frac[L], "alive_before": alive, "n_prune": k, "admit": bool(ok)}
            rec.update(info)
            if ok:
                ck = pr.save()
                if on_admit is not None:
                    rec.update(on_admit())
            else:
                pr.restore(ck)
                frac[L] /= 2.0
            rounds.append(rec)
            if log is not None:
                log(rec)
    return {"rounds": rounds, "frac_final": frac, "n_rounds": n, "hit_cap": n >= max_rounds}


# =============================================================================
# The surviving structure
# =============================================================================

def structure(w, arm, full_basis=None):
    """Per unit: which minted symbols it still reads (layer 0 / U, V) and writes (minted head), whether it is
    single-symbol, and how many distinct symbols survive across the net. For the one-hot layer 0 the surviving
    rows are read in the full recovered basis (projection, no constraint). k_dft is the logged oracle."""
    meta = arm["meta"]
    nsym = len(meta)
    out = {"sym_k_dft": [m["k_dft"] for m in meta]}
    if w.kind == "mlp":
        M0, M1, M2 = w.mask("layer0"), w.mask("layer1"), w.mask("head")
        alive0 = (M0.any(1) & M1.any(0)).numpy()
        alive1 = (M1.any(1) & M2.any(0)).numpy()
        units0 = []
        unit_sym = {}
        if w.L["layer0"]["kind"] == "cols":
            ga = w.L["layer0"]["galive"].view(-1, 2 * nsym).numpy()
            gn = w.group_norms("layer0").view(-1, 2 * nsym).numpy()
            for i in np.where(alive0)[0]:
                sa = [t for t in range(nsym) if ga[i, t]]
                sb = [t for t in range(nsym) if ga[i, nsym + t]]
                nd = sorted(set(sa + sb) - {0})
                nrm = {t: float(max(gn[i, t] if ga[i, t] else 0, gn[i, nsym + t] if ga[i, nsym + t] else 0))
                       for t in nd}
                dom = max(nd, key=lambda t: nrm[t]) if nd else 0
                unit_sym[int(i)] = dom
                units0.append({"u": int(i), "a": sa, "b": sb, "single": len(nd) == 1,
                               "same_ab": (set(sa) - {0}) == (set(sb) - {0}), "dom": int(dom)})
        else:
            W0 = w.L["layer0"]["w"].detach().double().numpy()
            Pf, cs = full_basis
            nfs = int(cs.max()) + 1
            for i in np.where(alive0)[0]:
                ea = np.array([((W0[i, :P] @ Pf[:, cs == t]) ** 2).sum() for t in range(nfs)])
                eb = np.array([((W0[i, P:] @ Pf[:, cs == t]) ** 2).sum() for t in range(nfs)])
                e = ea + eb
                dom = int(np.argmax(e[1:]) + 1)
                unit_sym[int(i)] = dom
                units0.append({"u": int(i), "n_res_a": int((W0[i, :P] != 0).sum()),
                               "n_res_b": int((W0[i, P:] != 0).sum()), "dom": dom,
                               "conc": float(e[dom] / (e[1:].sum() + 1e-300)),
                               "dc_frac": float(e[0] / (e.sum() + 1e-300))})
        out["units0"] = units0
        m1 = M1.numpy() > 0
        units1 = []
        for j in np.where(alive1)[0]:
            ins = [int(i) for i in np.where(m1[j] & alive0)[0]]
            syms = sorted(set(unit_sym[i] for i in ins))
            rec = {"u": int(j), "fan_in": len(ins), "in_syms": syms}
            if w.L["head"]["kind"] == "rows":
                gh = w.L["head"]["galive"].view(nsym, -1).numpy()
                rec["writes"] = [t for t in range(nsym) if gh[t, j]]
            else:
                rec["n_classes"] = int(M2[:, j].sum())
            units1.append(rec)
        out["units1"] = units1
        out["n_units0"], out["n_units1"] = len(units0), len(units1)
        out["read_syms"] = sorted(set(u["dom"] for u in units0)) if w.L["layer0"]["kind"] != "cols" else \
            sorted(set(t for u in units0 for t in u["a"] + u["b"]))
        if w.L["layer0"]["kind"] == "cols":
            out["n_single"] = int(sum(u["single"] for u in units0))
            out["n_same_ab"] = int(sum(u["same_ab"] for u in units0))
        out["n_l1_single_sym"] = int(sum(len(u["in_syms"]) == 1 for u in units1))
        out["l1_fan_in_median"] = float(np.median([u["fan_in"] for u in units1])) if units1 else None
        if w.L["head"]["kind"] == "rows":
            out["write_syms"] = sorted(set(t for u in units1 for t in u["writes"]))
            out["n_l1_writes_own_read"] = int(sum(set(u["writes"]) - {0} <= set(u["in_syms"]) for u in units1))
        per_sym = []
        for t in range(nsym):
            ps = {"sym": t, "k_dft": meta[t]["k_dft"] if t < len(meta) else None,
                  "n_units0": int(sum(u["dom"] == t for u in units0))}
            if w.L["layer0"]["kind"] == "cols":        # surviving (unit, operand) connections to symbol t
                ps["n_conn_a"] = int(sum(t in u["a"] for u in units0))
                ps["n_conn_b"] = int(sum(t in u["b"] for u in units0))
            per_sym.append(ps)
        out["per_sym"] = per_sym
    else:
        gU = w.L["U"]["galive"].view(-1, nsym - 1).numpy()
        gV = w.L["V"]["galive"].view(-1, nsym - 1).numpy()
        gW = w.L["W"]["galive"].view(nsym, -1).numpy()
        MU, MV, MW = w.mask("U"), w.mask("V"), w.mask("W")
        live = (MU.any(1) & MV.any(1) & MW.any(0)).numpy()
        units = []
        for i in np.where(live)[0]:
            su = [t + 1 for t in range(nsym - 1) if gU[i, t]]
            sv = [t + 1 for t in range(nsym - 1) if gV[i, t]]
            sw = [t for t in range(nsym) if gW[t, i]]
            units.append({"u": int(i), "U": su, "V": sv, "W": sw,
                          "single": len(set(su + sv + sw) - {0}) == 1})
        out["units"] = units
        out["n_units"] = len(units)
        out["n_single"] = int(sum(u["single"] for u in units))
        out["read_syms"] = sorted(set(t for u in units for t in u["U"] + u["V"]))
        out["write_syms"] = sorted(set(t for u in units for t in u["W"]))
    out["read_k_dft"] = sorted(set(meta[t]["k_dft"] for t in out["read_syms"] if t < len(meta)))
    return out


# =============================================================================
# One arm, end to end
# =============================================================================

def recover(w, target_chunks=RECOVER_CHUNKS):
    """Retrain in chunks of the per-round budget until train agreement is 1.0 (before the first prune)."""
    hist = []
    ev = w.eval()
    n = 0
    while ev["n_disagree"] > 0 and n < target_chunks:
        info = w.retrain()
        ev = w.eval()
        n += 1
        hist.append({"chunk": n, "epochs": info["epochs"], **ev})
    return {"chunks": n, "epochs": int(sum(h["epochs"] for h in hist)), "recovered": ev["n_disagree"] == 0,
            "hist": hist}


def init_checks(arm, w, snap, K, st):
    """G1: the arm at initialization against the net it rewrites."""
    out = {}
    with torch.no_grad():
        L_tr = w.model(w.x_tr).double()
        L_te = w.model(w.x_te).double()
        if arm["cfg"]["learner"] == "mlp":
            if arm["cfg"]["basis"] == "K":
                Kc = tuple(sorted(K))
                ref_tr, ref_te = snap.logits_C(Kc, "tr").double(), snap.logits_C(Kc, "te").double()
                out["ref"] = "form C (K)"
            else:
                ref_tr, ref_te = snap.L_tr.double(), snap.L_te.double()
                out["ref"] = "original net"
            if arm["cfg"]["head"] == "minted":        # the minted head is the reference's logits projected on span(Pm)
                Pi = torch.tensor(arm["Pm"] @ arm["Pm"].T)
                out["agree_unprojected_ref"] = float((ref_tr.argmax(1).numpy() == snap.own_tr).mean())
                ref_tr, ref_te = ref_tr @ Pi, ref_te @ Pi
            out["maxabs_vs_ref"] = float(max((L_tr - ref_tr).abs().max(), (L_te - ref_te).abs().max()))
            out["ref_heldout"] = float((ref_te.argmax(1).numpy() == snap.y_te_true).mean())
        else:
            a_tr, b_tr, a_te, b_te = snap.a_tr, snap.b_tr, snap.a_te, snap.b_te
            ref_tr = torch.tensor(arm["orig_fn"](a_tr, b_tr))
            ref_te = torch.tensor(arm["orig_fn"](a_te, b_te))
            out["ref"] = "banked bilinear (DFT input, plain head), projected on span(Pm)"
            Pi = torch.tensor(arm["Pm"] @ arm["Pm"].T)
            out["ref_train_agree_unprojected"] = float((ref_tr.argmax(1).numpy() == snap.own_tr).mean())
            ref_tr, ref_te = ref_tr @ Pi, ref_te @ Pi
            out["maxabs_vs_ref"] = float(max((L_tr - ref_tr).abs().max(), (L_te - ref_te).abs().max()))
            out["ref_heldout"] = float((ref_te.argmax(1).numpy() == snap.y_te_true).mean())
            out["ref_train_agree"] = float((ref_tr.argmax(1).numpy() == snap.own_tr).mean())
    out.update(w.eval())
    return out


def run_arm(name, data_dir=DATA_DIR, retrain=RETRAIN, f0=F0, floor=FLOOR, max_rounds=MAX_ROUNDS,
            loaded=None, verbose=True, save_dir=None, settle=SETTLE):
    t0 = time.time()
    st, snap, K, split = loaded if loaded is not None else load_final(data_dir)
    cfg = ARMS[name]
    if "retrain" in cfg:
        retrain = cfg["retrain"] if retrain >= RETRAIN else min(retrain, cfg["retrain"])   # smoke keeps its cap
    bil = None
    if cfg["learner"] == "bilinear":
        z = np.load(os.path.join(data_dir, *BIL_BANK))
        bil = {k: z[k] for k in z.files}
    arm = build_arm(name, st, snap, K, split, bil_state=bil)
    y_own = snap.own_tr                                   # the original net's train argmax, banked here
    assert (y_own == snap.y_tr_true).all()
    w = Whittler(arm, y_own, split, cfg["wd"], retrain, settle)
    rec = {"arm": name, "cfg": cfg, "K": list(K), "cand": arm["cand"], "meta": arm["meta"],
           "basis_info": arm["basis_info"], "retrain": retrain, "settle": settle, "f0": f0, "floor": floor,
           "max_rounds": max_rounds, "s_feat": S_FEAT}
    rec["init"] = init_checks(arm, w, snap, K, st)
    rec["init"].update(w.counts())
    rec["init"]["probe"] = w.probe()
    rec["n_params_init"] = int(sum(p.numel() for p in w.model.parameters()))
    if verbose:
        print(f"[{name}] init: {json.dumps({k: v for k, v in rec['init'].items() if k != 'probe'})}", flush=True)
    rec["recover"] = recover(w) if rec["init"]["n_disagree"] > 0 else None
    if rec["recover"] is not None and verbose:
        print(f"[{name}] recover: {rec['recover']['chunks']} chunks, recovered={rec['recover']['recovered']}",
              flush=True)
    rec["start"] = dict(w.eval(), **w.counts())
    rec["start"]["probe"] = w.probe()
    ck_start = w.save()

    def on_admit():
        return {"counts": w.counts(), "probe": {k: v for k, v in w.probe().items() if k != "alpha"}}

    def log(r):
        if verbose:
            c = r.get("counts", {})
            print(f"  [{name}] r{r['round']:3d} {r['layer']:>6s} f={r['frac']:.4f} -{r['n_prune']:5d} "
                  f"{'ADMIT' if r['admit'] else 'reject'} dis={r['n_disagree']:4d} ho={r['heldout']:.4f} "
                  f"post={r['agree_post_prune']:.4f} ok@{r['first_ok_epoch']}/{r['epochs']} w={c.get('weights', '-')} "
                  f"u={c.get('units0', c.get('units', '-'))}/{c.get('units1', '-')} "
                  f"gm={r.get('probe', {}).get('gbar_margin', float('nan')):.3f} t={time.time() - t0:.0f}s",
                  flush=True)

    walk = whittle_walk(w, w.order, f0=f0, floor=floor, max_rounds=max_rounds, on_admit=on_admit, log=log)
    rec["walk"] = walk
    rec["final"] = dict(w.eval(), **w.counts())
    rec["final"]["probe"] = w.probe()
    rec["final"]["logit_energy"] = w.logit_energy()
    Pf, cs, _ = minted_basis(snap, [int(j) for j in snap.producer_order()])
    rec["structure"] = structure(w, arm, full_basis=(Pf, cs))
    rec["full_basis_k_dft"] = [0] + [int(snap.read["k_match"][j]) for j in snap.producer_order()]
    rec["stats"] = dict(w.stats)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        arrs = {f"param/{k}": v.detach().numpy() for k, v in w.model.state_dict().items()}
        arrs.update({f"galive/{n}": w.L[n]["galive"].numpy() for n in w.order})
        arrs.update({f"bmask/{b}": m.numpy() for b, m in w.bmask.items()})
        arrs["Pm"] = arm["Pm"]
        np.savez_compressed(os.path.join(save_dir, f"{name}_final.npz"), **arrs)
    # drift twin: the start state, nothing pruned, retrained with the admitted rounds' epoch counts in order
    # (fresh AdamW per chunk, no early stop), held-out logged after each chunk. Separates retraining drift
    # from pruning in the held-out series. Logged only.
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
        print(f"[{name}] final: {json.dumps({k: v for k, v in rec['final'].items() if k not in ('probe', 'logit_energy')})} "
              f"rounds={walk['n_rounds']} {rec['seconds']:.0f}s stats={rec['stats']} rss={rec['peak_rss_mb']:.0f}MB",
              flush=True)
    return rec


# =============================================================================
# Gates
# =============================================================================

class MockPruner:
    """G4: a designed sequence. Layer L has n_L groups; the gate passes iff pruned_L <= cap_L for every L
    (optionally also failing on a scripted list of round numbers, to exercise restore)."""

    def __init__(self, n, cap, fail_rounds=()):
        self.n, self.cap = dict(n), dict(cap)
        self.pruned = {L: 0 for L in n}
        self.fail_rounds = set(fail_rounds)
        self.r = 0

    def n_alive(self, L):
        return self.n[L] - self.pruned[L]

    def prune(self, L, k):
        self.pruned[L] += k

    def settle(self):
        self.r += 1
        ok = all(self.pruned[L] <= self.cap[L] for L in self.n) and self.r not in self.fail_rounds
        return ok, {"pruned": dict(self.pruned)}

    def save(self):
        return dict(self.pruned)

    def restore(self, s):
        self.pruned = dict(s)


def gate_G4():
    out = {}
    m = MockPruner({"A": 100, "B": 50, "C": 7}, {"A": 73, "B": 20, "C": 0})
    r = whittle_walk(m, ["A", "B", "C"], f0=0.5, floor=0.005, max_rounds=1000)
    out["G4a_final"] = dict(m.pruned)
    out["G4a_expected"] = {"A": 73, "B": 20, "C": 0}
    out["G4a_rounds"] = r["n_rounds"]
    out["G4a_pass"] = m.pruned == {"A": 73, "B": 20, "C": 0}
    out["G4a_trace_A"] = [(x["n_prune"], x["admit"]) for x in r["rounds"] if x["layer"] == "A"]
    # scripted spurious failures: the walk must restore and still end at the caps (it may end below them if a
    # spurious failure halves a fraction past the floor; the cap is never exceeded)
    m2 = MockPruner({"A": 100, "B": 50}, {"A": 73, "B": 20}, fail_rounds=(1, 2, 5))
    r2 = whittle_walk(m2, ["A", "B"], f0=0.5, floor=0.005, max_rounds=1000)
    out["G4b_final"] = dict(m2.pruned)
    out["G4b_never_exceeds"] = all(m2.pruned[L] <= m2.cap[L] for L in m2.n)
    out["G4b_restores"] = all((not x["admit"]) or all(v <= m2.cap[L] for L, v in x["pruned"].items())
                              for x in r2["rounds"])
    # the cap on rounds
    m3 = MockPruner({"A": 100}, {"A": 100})
    r3 = whittle_walk(m3, ["A"], f0=0.5, floor=0.005, max_rounds=3)
    out["G4c_rounds"], out["G4c_hit_cap"] = r3["n_rounds"], r3["hit_cap"]
    out["pass"] = bool(out["G4a_pass"] and out["G4b_never_exceeds"] and out["G4b_restores"] and r3["n_rounds"] == 3)
    return out


def gates_all(data_dir=DATA_DIR, retrain=RETRAIN, arms=("committed", "unrestricted", "both_sides", "all48",
                                                        "committed_wd1", "bilinear_both")):
    t0 = time.time()
    loaded = load_final(data_dir)
    st, snap, K, split = loaded
    res = {"K": list(K), "K_dft": snap.kmatch(K), "K13": K13, "K_dft_equals_K13": snap.kmatch(K) == sorted(K13),
           "form_C_heldout": snap.heldout("C", tuple(sorted(K)))}
    z = np.load(os.path.join(data_dir, *BIL_BANK))
    bil = {k: z[k] for k in z.files}
    for name in arms:
        cfg = ARMS[name]
        g = {}
        # G1: init reproduces the net it rewrites
        arm = build_arm(name, st, snap, K, split, bil_state=bil)
        w = Whittler(arm, snap.own_tr, split, cfg["wd"], retrain)
        g["G1"] = init_checks(arm, w, snap, K, st)
        g["basis_info"] = arm["basis_info"]
        g["n_params"] = int(sum(p.numel() for p in w.model.parameters()))
        g["counts"] = w.counts()
        # G3: zero-prune retrain at the per-round budget (both_sides / bilinear after recovery)
        rc = recover(w) if g["G1"]["n_disagree"] > 0 else None
        g["recover"] = rc
        w.retrain(settle=0)
        g["G3"] = w.eval()
        g["G3"]["pass"] = bool(g["G3"]["n_disagree"] == 0 and g["G3"]["heldout"] >= 0.999)
        # G2: prune half of every layer, cascade, retrain at wd 0 and wd 1: pruned entries stay exactly zero;
        # G2b: the cascade is function-preserving
        for wd in (0.0, 1.0):
            arm2 = build_arm(name, st, snap, K, split, bil_state=bil)
            w2 = Whittler(arm2, snap.own_tr, split, wd, retrain)
            for n in w2.order:
                w2.prune(n, w2.n_alive(n) // 2)
            with torch.no_grad():
                Lb = w2.model(w2.x_tr).clone()
            w2.cascade()
            with torch.no_grad():
                La = w2.model(w2.x_tr)
            w2.retrain(epochs=200, settle=0)
            pm = _pmap(w2.model)
            zero_ok = all(bool((pm[f"{n}.weight"][w2.mask(n) == 0] == 0).all()) for n in w2.order)
            bz = all(bool((pm[f"{b}.bias"][m == 0] == 0).all()) for b, m in w2.bmask.items())
            g[f"G2_wd{wd:g}"] = {"pruned_exact_zero": zero_ok, "dead_bias_zero": bz,
                                 "cascade_maxabs": float((La - Lb).abs().max()),
                                 "counts_after": w2.counts(), "folds": [w2.stats["n_fold0"], w2.stats["n_fold1"]]}
        res[name] = g
        print(f"  [{name}] G1 {g['G1']} | G3 {g['G3']} | G2 {g['G2_wd0']['pruned_exact_zero']},"
              f"{g['G2_wd1']['pruned_exact_zero']} cascade {g['G2_wd0']['cascade_maxabs']:.2e} "
              f"| {time.time() - t0:.0f}s", flush=True)
    # G2c: a designed cascade on the committed net: silence all incoming weights of 5 layer-0 units (constant
    # units), all outgoing of 5 others (dead ends), all head weights of 3 layer-1 units; function preserved.
    arm = build_arm("committed", st, snap, K, split)
    w = Whittler(arm, snap.own_tr, split, 0.0, retrain)
    with torch.no_grad():
        Lb = w.model(w.x_tr).clone()
        M0 = w.mask("layer0")
        M0[:5, :] = 0
        w._set_from_entry_mask("layer0", M0)
        M1 = w.mask("layer1")
        M1[:, 5:10] = 0
        w._set_from_entry_mask("layer1", M1)
        M2 = w.mask("head")
        M2[:, :3] = 0
        w._set_from_entry_mask("head", M2)
        # zeroing weights changes the function; the reference is the net right after the zeroing
        w.apply_masks()
        Lz = w.model(w.x_tr).clone()
        w.cascade()
        La = w.model(w.x_tr)
    res["G2c_cascade"] = {"maxabs_vs_zeroed": float((La - Lz).abs().max()), "counts": w.counts(),
                          "folds": [w.stats["n_fold0"], w.stats["n_fold1"]]}
    res["G4"] = gate_G4()
    res["seconds"] = time.time() - t0
    res["peak_rss_mb"] = peak_rss_mb()
    return res


# =============================================================================
# Modal entrypoints (names prefixed `whittle_`: mint / basis / rung functions share the app)
# =============================================================================

@app.function(cpu=4.0, memory=2048, timeout=1800, volumes={DATA_DIR: volume})
def whittle_gates(retrain: int = RETRAIN):
    volume.reload()
    res = gates_all(DATA_DIR, retrain)
    print(f"G2c {res['G2c_cascade']}", flush=True)
    print(f"G4 {res['G4']}", flush=True)
    print(f"gates {res['seconds']:.0f}s rss={res['peak_rss_mb']:.0f}MB", flush=True)
    d = os.path.join(DATA_DIR, "whittle_gates")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"gates_R{retrain}.json"), "w") as f:
        json.dump(res, f)
    volume.commit()
    return {k: res[k] for k in ("K", "K_dft", "K_dft_equals_K13", "G4", "seconds")}


@app.function(cpu=4.0, memory=1536, timeout=3 * 3600, volumes={DATA_DIR: volume}, max_containers=6)
def whittle_arm(tag: str, name: str, retrain: int, f0: float, floor: float, max_rounds: int, settle: int):
    volume.reload()
    d = os.path.join(DATA_DIR, tag)
    rec = run_arm(name, DATA_DIR, retrain, f0, floor, max_rounds, save_dir=d, settle=settle)
    with open(os.path.join(d, f"{name}.json"), "w") as f:
        json.dump(rec, f)
    volume.commit()
    return rec


@app.function(cpu=1.0, memory=1024, timeout=4 * 3600, volumes={DATA_DIR: volume})
def whittle_run(tag: str = "w1", arms: str = ",".join(MAIN_ARMS + ("bilinear_both", "committed_R2000")), smoke: int = 0,
                retrain: int = RETRAIN, f0: float = F0, floor: float = FLOOR, max_rounds: int = MAX_ROUNDS,
                settle: int = SETTLE):
    """Coordinator (CPU only): one container per arm via starmap; writes /data/<tag>/whittle.json."""
    t0 = time.time()
    names = [a for a in arms.split(",") if a]
    if smoke:
        retrain, max_rounds, settle = min(retrain, 200), min(max_rounds, 9), min(settle, 20)
    print(f"{len(names)} arms {names} retrain={retrain} settle={settle} f0={f0} floor={floor} "
          f"max_rounds={max_rounds}", flush=True)
    recs = list(whittle_arm.starmap([(tag, n, retrain, f0, floor, max_rounds, settle) for n in names]))
    out = {"tag": tag, "arms": names, "smoke": smoke, "retrain": retrain, "settle": settle, "f0": f0, "floor": floor,
           "max_rounds": max_rounds, "results": {r["arm"]: r for r in recs}, "seconds": time.time() - t0}
    d = os.path.join(DATA_DIR, tag)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "whittle.json")
    with open(path, "w") as f:
        json.dump(out, f)
    volume.commit()
    print(f"whittle done in {time.time() - t0:.0f}s -> {path}; per arm: "
          f"{[(r['arm'], r['final']['weights'], round(r['final']['heldout'], 4), round(r['seconds'])) for r in recs]}",
          flush=True)
    return path
