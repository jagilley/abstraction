"""[grokking/basis] Endogenous basis mint: read the net's own operator a -> net(a, g), diagonalize
it, and walk / compile over the recovered subspaces instead of the gifted DFT basis.

No retraining: every net is a banked m1 snapshot on `grokking-mint-data` (/data/<mint_tag>/{true,shuffled}/
snapshots.npz) or an untrained init rebuilt from its seed. At each snapshot:

  1. READ THE OPERATOR. For each generator g in GENS (N_GEN symbols drawn once with a fixed seed), the
     transition matrix M_g[c, a] = softmax(net(a, g))[c] over all p symbols: the net's own predictions,
     no labels. Beside it, the hard map a -> argmax_c net(a, g): bijection flag, cycle count, fixed points.
  2. DIAGONALIZE M_{g_ref} (g_ref = GENS[0]). The eigenvector at eigenvalue 1 (M is column-stochastic) is
     the recovered DC. Each complex-conjugate eigenpair gives a real 2-D invariant subspace span{Re v, Im v};
     each other real eigenvalue gives a 1-D subspace. These are the candidate units. Eigenvalue phase and
     modulus are logged, and so is the restricted operator Q^T M_g Q on every candidate for every g.
  3. ENDOGENOUS CERTIFICATE (no labels): cross-g agreement. For each candidate, its best projector overlap
     with the candidates of each other M_g, minimum over g. Also the commutator norm of the M_g.
  4. LOGGED ORACLE, never consumed: projector overlap of each candidate with the true DFT character pair
     at the best-matching frequency (k_match), and the restricted operator's phase at every g against the
     true character value at g.
  5. WALK AND COMPILE, as in m1 (`mint.py`). Producer score = the net's layer-0 a-row and b-row energy
     projected onto each candidate (times p, so a DFT pair gives m1's |F_w|^2 + |F_{p-w}|^2). census_walk
     runs in producer order under le0 / lt1 / m3, with one gate per form: the form's agreement with the
     net's own train argmax. The forms are compiled over the RECOVERED basis:
       A    logits[c] = sum_{j in K} Re[psi_j(a) psi_j(b) conj(psi_j(c))]. psi_j is the eigenvector
            phase-fixed at the net's own identity symbol e_hat (the b-slot symbol s maximising
            mean_a softmax(net(a, s))[a]; no labels) and scaled to rms 1. With exact characters this is m1's
            form A, sum_w cos 2 pi w (a+b-c)/p, so the m1 degeneracy (argmax a+b at every nonempty K)
            carries over unchanged.
       ALS  the same terms with amplitudes, fit by joint least squares (per-row intercepts) to the net's
            train logits, refit at every K. With exact characters the design is orthogonal and this equals
            m1's per-frequency cosine coefficient.
       B    the net with every hidden unit whose dominant candidate is not in K zeroed (no retraining).
       C    every layer-0 row orthogonally projected onto span(recovered DC + candidates in K).
     Held-out label accuracy is logged beside, never consumed.

Falsifiers: the shuffled-label net and the untrained inits go through the identical pipeline. Designed
checks (`designed_checks`): an exact 97-cycle (the shift) must recover the characters to numerical
precision; a random permutation and a random 97-cycle must not; a relabelled group action must pass the
certificate but fail the DFT gauge; a soft circulant must recover the characters exactly. Instrument check:
the exact DFT basis injected into this pipeline reproduces m1's recorded K and held-out.

Commands (run from experiments/, MODAL_PROFILE=chromatic):
    modal run grokking/basis/basis.py::basis_gates
    modal run grokking/basis/basis.py::basis_run --tag bsmoke --smoke 1
    modal run --detach grokking/basis/basis.py::basis_run --tag b1
    python3 grokking/basis/reduce_basis.py --tag b1 --fetch
"""

import json
import math
import os
import time

import numpy as np

from grokking.shared import (DATA_DIR, HIDDEN, P, SEED, F, GrokMLP, app, onehot, peak_rss_mb,
                                  split_pairs, torch, volume)
from grokking.mint import (FORMS, INIT_SEEDS, N_RAND, RAND_SEED0, Snapshot, _flat, _load_snaps,
                                _run_dir, census_walk, gate_rules)

N_FREQ = (P - 1) // 2
N_GEN = 6
GEN_SEED = SEED
# symbols only as names: drawn once, fixed across snapshots. The draw is (62, 40, 93, 18, 81, 83); it does
# not contain symbol 0 (the identity), whose operator would be I with fully degenerate eigenspaces.
GENS = tuple(int(x) for x in np.random.RandomState(GEN_SEED).choice(P, N_GEN, replace=False))


# =============================================================================
# Pure linear algebra (numpy only; importable without torch)
# =============================================================================

def orth(B, rtol=1e-10):
    """Orthonormal basis of span(columns of B), rank-revealing (SVD)."""
    B = np.asarray(B, np.float64)
    if B.ndim == 1:
        B = B[:, None]
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return U[:, :0]
    return U[:, s > rtol * s[0]]


def fold_angle(t):
    t = np.mod(t, 2 * np.pi)
    return np.minimum(t, 2 * np.pi - t)


def dft_pairs(p=P):
    """The true character pairs (the oracle): orthonormal real bases C_k (p, 2), k = 1..(p-1)/2."""
    a = np.arange(p)
    return [orth(np.stack([np.cos(2 * np.pi * k * a / p), np.sin(2 * np.pi * k * a / p)], 1))
            for k in range(1, (p - 1) // 2 + 1)]


def decompose(M):
    """Real invariant subspaces of a real operator M (p x p).

    Returns {"cands": [{"lam", "vec", "Q"}], "dc", "lam0", "n_pairs", "n_real"}. The DC is the real
    eigenvector whose eigenvalue is closest to 1 (1 is an eigenvalue of any column-stochastic M). Candidates:
    one per complex-conjugate pair (the Im > 0 member; Q = orth[Re v, Im v], 2-D), ordered by phase, then
    one per other real eigenvalue (1-D), ordered by eigenvalue descending."""
    lam, V = np.linalg.eig(np.asarray(M, np.float64))
    lam = lam.astype(complex)
    V = V.astype(complex)
    real = np.where(lam.imag == 0)[0]
    i0 = int(real[np.argmin(np.abs(lam[real].real - 1.0))])
    pairs = np.where(lam.imag > 0)[0]
    pairs = pairs[np.argsort(np.angle(lam[pairs]), kind="stable")]
    reals = sorted([int(i) for i in real if i != i0], key=lambda i: -lam[i].real)
    cands = []
    for i in pairs:
        v = V[:, i]
        cands.append({"lam": complex(lam[i]), "vec": v, "Q": orth(np.stack([v.real, v.imag], 1))})
    for i in reals:
        v = V[:, i].real
        cands.append({"lam": complex(lam[i]), "vec": v.astype(complex), "Q": orth(v)})
    u0 = V[:, i0].real
    u0 = u0 / (np.linalg.norm(u0) + 1e-300)
    return {"cands": cands, "dc": u0, "lam0": float(lam[i0].real), "n_pairs": int(len(pairs)),
            "n_real": int(len(real))}


def overlap_matrix(Qs1, Qs2):
    """O[j, l] = ||Q1_j^T Q2_l||_F^2 / max(dim_j, dim_l): projector overlap, 1 iff the subspaces are equal."""
    if not Qs1 or not Qs2:
        return np.zeros((len(Qs1), len(Qs2)))
    d1 = np.array([q.shape[1] for q in Qs1])
    d2 = np.array([q.shape[1] for q in Qs2])
    S = (np.concatenate(Qs1, 1).T @ np.concatenate(Qs2, 1)) ** 2
    i1 = np.repeat(np.arange(len(Qs1)), d1)
    i2 = np.repeat(np.arange(len(Qs2)), d2)
    O = np.zeros((len(Qs1), len(Qs2)))
    np.add.at(O, (i1[:, None], i2[None, :]), S)
    return O / np.maximum(d1[:, None], d2[None, :])


def restricted_phase(Q, M):
    """Eigenvalue of the operator restricted to span(Q): Q^T M Q. For a 2-D candidate, the member with
    Im >= 0 (phase in [0, pi]); for 1-D, the scalar. Returns (phase, modulus)."""
    B = Q.T @ M @ Q
    mu = np.linalg.eigvals(B)
    mu = mu[np.argmax(mu.imag)] if len(mu) > 1 else mu[0]
    return float(abs(np.angle(mu))), float(abs(mu))


def read_operators(Ms, gens, dft_Q=None, p=P, dec_ref=None, decs=None):
    """Steps 2-4 on a dict {g: M_g}. Returns (dec_ref, rec). rec holds per-candidate arrays and summaries.
    dec_ref may be injected (e.g. the exact DFT basis, for the instrument check)."""
    dft_Q = dft_pairs(p) if dft_Q is None else dft_Q
    g_ref = gens[0]
    if decs is None:
        decs = {g: decompose(Ms[g]) for g in gens}
    if dec_ref is None:
        dec_ref = decs[g_ref]
    cands = dec_ref["cands"]
    Qs = [c["Q"] for c in cands]
    m = len(cands)
    dims = np.array([q.shape[1] for q in Qs])
    # step 3: cross-g agreement (endogenous)
    xg_by_g = {}
    xg = np.ones(m)
    for g in gens[1:]:
        O = overlap_matrix(Qs, [c["Q"] for c in decs[g]["cands"]])
        best = O.max(1) if O.size else np.zeros(m)
        xg_by_g[str(g)] = float(best.mean()) if m else float("nan")
        xg = np.minimum(xg, best)
    comm = []
    for i, g in enumerate(gens):
        for h in gens[i + 1:]:
            A, B = Ms[g], Ms[h]
            comm.append(float(np.linalg.norm(A @ B - B @ A) / (np.linalg.norm(A @ B) + 1e-300)))
    # step 4: DFT oracle
    Od = overlap_matrix(Qs, dft_Q)
    dft_ov = Od.max(1) if m else np.zeros(0)
    k_match = (Od.argmax(1) + 1) if m else np.zeros(0, int)
    dft_rec = Od.max(0) if m else np.zeros(len(dft_Q))
    dc_ov = float((dec_ref["dc"].sum()) ** 2 / p)
    # phases: restricted operator on every candidate for every g; oracle = true character value at g
    theta = np.zeros((m, len(gens)))
    modulus = np.zeros((m, len(gens)))
    for j, Q in enumerate(Qs):
        for gi, g in enumerate(gens):
            theta[j, gi], modulus[j, gi] = restricted_phase(Q, Ms[g])
    true_theta = np.array([[fold_angle(2 * np.pi * k * g / p) for g in gens] for k in k_match]) \
        if m else np.zeros((0, len(gens)))
    phase_err = np.abs(theta - true_theta)
    lam = np.array([c["lam"] for c in cands]) if m else np.zeros(0, complex)
    quant = np.abs(p * np.abs(np.angle(lam)) / (2 * np.pi) - np.round(p * np.abs(np.angle(lam)) / (2 * np.pi)))
    pair = dims == 2
    rec = {
        "m": int(m), "n_pairs": dec_ref["n_pairs"], "n_real": dec_ref["n_real"], "lam0": dec_ref["lam0"],
        "cand_dim": dims.tolist(),
        "lam_mod": np.abs(lam).round(6).tolist(), "lam_phase": np.abs(np.angle(lam)).round(6).tolist(),
        "quant_resid": quant.round(5).tolist(),
        "xg": xg.round(6).tolist(), "xg_by_g": xg_by_g,
        "xg_mean": float(xg.mean()) if m else float("nan"), "xg_min": float(xg.min()) if m else float("nan"),
        "xg_frac99": float((xg > 0.99).mean()) if m else float("nan"),
        "comm_mean": float(np.mean(comm)) if comm else float("nan"),
        "dft_ov": dft_ov.round(6).tolist(), "k_match": [int(k) for k in k_match],
        "dft_rec": dft_rec.round(6).tolist(),
        "dft_ov_mean": float(dft_ov.mean()) if m else float("nan"),
        "dft_ov_min": float(dft_ov.min()) if m else float("nan"),
        "dft_rec_min": float(dft_rec.min()), "dft_rec_mean": float(dft_rec.mean()),
        "n_kmatch_distinct": int(len(set(int(k) for k in k_match))),
        "one_to_one": bool(m == len(dft_Q) and len(set(int(k) for k in k_match)) == len(dft_Q)),
        "dc_ov": dc_ov,
        "theta": theta.round(6).tolist(), "theta_mod": modulus.round(6).tolist(),
        "phase_err": phase_err.round(7).tolist(),
        "phase_err_max": float(phase_err[pair].max()) if pair.any() else float("nan"),
        "phase_err_median": float(np.median(phase_err[pair])) if pair.any() else float("nan"),
    }
    return dec_ref, rec


def dft_decomposition(p=P):
    """The exact DFT basis in `decompose`'s format (for the instrument check): DC = ones/sqrt(p)."""
    a = np.arange(p)
    Qs = dft_pairs(p)
    cands = [{"lam": complex(np.nan, np.nan), "vec": np.exp(2j * np.pi * k * a / p), "Q": Qs[k - 1]}
             for k in range(1, (p - 1) // 2 + 1)]
    return {"cands": cands, "dc": np.ones(p) / np.sqrt(p), "lam0": 1.0, "n_pairs": len(cands), "n_real": 1}


# --- designed operators ----------------------------------------------------------------------------

def perm_matrix(pi):
    """M[pi(a), a] = 1: the operator of the map a -> pi(a) (column-stochastic)."""
    p = len(pi)
    M = np.zeros((p, p))
    M[np.asarray(pi), np.arange(p)] = 1.0
    return M


def shift_perm(g, p=P):
    return (np.arange(p) + g) % p


def n_cycles(pi):
    pi = np.asarray(pi)
    seen = np.zeros(len(pi), bool)
    n = 0
    for s in range(len(pi)):
        if not seen[s]:
            n += 1
            x = s
            while not seen[x]:
                seen[x] = True
                x = pi[x]
    return n


def designed_checks(gens=GENS, p=P, seed=0):
    """The designed operators through steps 2-4. Returns {name: summary}."""
    rng = np.random.RandomState(seed)
    dQ = dft_pairs(p)
    keys = ("m", "n_pairs", "n_real", "xg_mean", "xg_min", "dft_ov_mean", "dft_ov_min", "dft_rec_min",
            "n_kmatch_distinct", "one_to_one", "phase_err_max", "dc_ov", "comm_mean")
    out = {}

    def summ(Ms, extra=None):
        _, r = read_operators(Ms, gens, dQ, p)
        s = {k: r[k] for k in keys}
        s["lam_mod_min"] = float(min(r["lam_mod"])) if r["lam_mod"] else float("nan")
        s["quant_max"] = float(max(r["quant_resid"])) if r["quant_resid"] else float("nan")
        if extra:
            s.update(extra(r))
        return s

    # D-1 the exact 97-cycle, the shift a -> a+g in the symbols' own labels
    out["D1_shift"] = summ({g: perm_matrix(shift_perm(g, p)) for g in gens})
    # D-2 a random permutation per g (generic cycle structure), independent across g
    pis = {g: rng.permutation(p) for g in gens}
    out["D2_random_perm"] = summ({g: perm_matrix(pis[g]) for g in gens})
    out["D2_random_perm"]["n_cycles"] = [n_cycles(pis[g]) for g in gens]
    # D-2b a random single 97-cycle per g, independent across g
    cyc = {}
    for g in gens:
        order = rng.permutation(p)
        pi = np.empty(p, int)
        pi[order] = np.roll(order, -1)
        cyc[g] = pi
    out["D2b_random_cycle"] = summ({g: perm_matrix(cyc[g]) for g in gens})
    out["D2b_random_cycle"]["n_cycles"] = [n_cycles(cyc[g]) for g in gens]
    # D-3 a relabelled group action: sigma o shift_g o sigma^-1 (common sigma). Certificate must pass,
    # DFT gauge (integer labels) must fail, and the relabelled characters must be recovered.
    sigma = rng.permutation(p)
    sinv = np.argsort(sigma)
    Ms3 = {g: perm_matrix(sigma[shift_perm(g, p)[sinv]]) for g in gens}
    dQ_rel = [q[sinv] for q in dQ]            # chi_k o sigma^-1

    def rel(r, Ms=Ms3):
        dec = decompose(Ms[gens[0]])
        O = overlap_matrix([c["Q"] for c in dec["cands"]], dQ_rel)
        return {"relabelled_dft_ov_min": float(O.max(1).min())}
    out["D3_relabelled_action"] = summ(Ms3, rel)
    # D-4 a soft shift, the form a confident grokked net approximates: M_g = 0.7 P_g + 0.3 U_g, U_g the
    # circulant with an even random kernel u = softmax(f). Its eigenvalues 0.7 + 0.3 u_hat(k) > 0 times
    # chi_k(g)^-1, so the subspaces AND the phases are exact.
    alpha = rng.randn(N_FREQ)
    d = np.arange(p)
    f = sum(alpha[k - 1] * np.cos(2 * np.pi * k * d / p) for k in range(1, N_FREQ + 1))
    u = np.exp(f - f.max())
    u = u / u.sum()

    def soft(g):
        a = np.arange(p)[None, :]
        c = np.arange(p)[:, None]
        return 0.7 * perm_matrix(shift_perm(g, p)) + 0.3 * u[(a + g - c) % p]
    out["D4_soft_circulant"] = summ({g: soft(g) for g in gens})
    return out


# =============================================================================
# One snapshot, the recovered basis in place of the DFT (subclass of m1's Snapshot)
# =============================================================================

def _softmax(L, axis):
    L = L - L.max(axis, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis, keepdims=True)


class BasisSnapshot(Snapshot):
    """m1's Snapshot (net, pools, own argmax, tie-aware agreement, B's pruning, aud cache) with the
    recovered basis in place of the DFT: `dom`, `score`, `support` and forms A / ALS / C are recomputed over
    the candidates of M_{g_ref}. m1's DFT read is kept as `dom_dft`, `score_dft` for comparison.
    `basis="dft"` injects the exact DFT basis (instrument check: must reproduce m1)."""

    def __init__(self, state, split, train_labels=None, gens=GENS, basis="recovered", p=P):
        super().__init__(state, split, train_labels=train_labels, p=p)
        self.dom_dft, self.score_dft, self.support_dft = self.dom, self.score, self.support
        self.gens = tuple(gens)
        # --- the full table, net's own softmax over all p^2 inputs (no labels) ---
        A_all = np.repeat(np.arange(p), p)
        B_all = np.tile(np.arange(p), p)
        with torch.no_grad():
            L_all = self.net(onehot(A_all, B_all, p)).double().numpy().reshape(p, p, p)   # [a, b, c]
        S = _softmax(L_all, 2)
        ar = np.arange(p)
        self.id_score = S[ar, :, ar].mean(0)                    # id_score[s] = mean_a softmax(net(a,s))[a]
        self.e_hat = int(np.argmax(self.id_score))
        # --- step 1: operators ---
        self.Ms, self.ops = {}, {}
        for g in self.gens:
            self.Ms[g] = S[:, g, :].T.copy()                     # M_g[c, a]
            hard = L_all[:, g, :].argmax(1)                      # a -> c
            bij = len(set(hard.tolist())) == p
            self.ops[str(g)] = {
                "bijection": bool(bij), "n_image": int(len(set(hard.tolist()))),
                "n_cycles": int(n_cycles(hard)) if bij else None,
                "n_fixed": int((hard == ar).sum()),
                "conf": float(S[:, g, :].max(1).mean()),
                "shift_acc_hard": float((hard == (ar + g) % p).mean()),       # oracle
                "shift_mass_soft": float(S[ar, g, (ar + g) % p].mean())}      # oracle
        # --- steps 2-4 ---
        inj = dft_decomposition(p) if basis == "dft" else None
        self.dec, self.read = read_operators(self.Ms, self.gens, None, p, dec_ref=inj)
        cands = self.dec["cands"]
        self.m = len(cands)
        self.Qs = [c["Q"] for c in cands]
        self.u0 = self.dec["dc"]
        # --- step 5a: producer read over the recovered candidates ---
        W0 = np.asarray(state["layer0.weight"], np.float64)
        self.Wa, self.Wb = W0[:, :p], W0[:, p:2 * p]
        Ea = np.stack([p * ((self.Wa @ Q) ** 2).sum(1) for Q in self.Qs], 1)
        Eb = np.stack([p * ((self.Wb @ Q) ** 2).sum(1) for Q in self.Qs], 1)
        self.E = Ea + Eb
        self.dom = np.argmax(self.E, axis=1)                     # candidate index, DC excluded
        self.score = self.E.sum(0)
        self.support = np.bincount(self.dom, minlength=self.m)
        self.dc_energy_rec = float(p * (((self.Wa @ self.u0) ** 2).sum() + ((self.Wb @ self.u0) ** 2).sum()))
        # --- step 5b: forms A / ALS over phase-fixed eigenvectors ---
        Psi = np.zeros((p, self.m), complex)
        for j, c in enumerate(cands):
            v = np.asarray(c["vec"], complex)
            ve = v[self.e_hat]
            ph = ve / abs(ve) if abs(ve) > 0 else 1.0
            v = v * np.conj(ph)
            if c["Q"].shape[1] == 1:
                v = v.real.astype(complex)
            Psi[:, j] = v / (np.sqrt(np.mean(np.abs(v) ** 2)) + 1e-300)
        self.Psi = Psi
        self.Ztr = Psi[self.a_tr] * Psi[self.b_tr]
        self.Zte = Psi[self.a_te] * Psi[self.b_te]
        # ALS: joint LS with per-row intercepts == LS on row-centred features and targets
        Lc = self.L_tr.double().numpy()
        Lc = Lc - Lc.mean(1, keepdims=True)
        Pc = Psi - Psi.mean(0, keepdims=True)
        Ar, Ai, Br, Bi = self.Ztr.real, self.Ztr.imag, Pc.real, Pc.imag
        self.als_b = (Ar * (Lc @ Br)).sum(0) + (Ai * (Lc @ Bi)).sum(0)
        self.als_G = ((Ar.T @ Ar) * (Br.T @ Br) + (Ar.T @ Ai) * (Br.T @ Bi)
                      + (Ai.T @ Ar) * (Bi.T @ Br) + (Ai.T @ Ai) * (Bi.T @ Bi))
        self.alpha_cache = {}

    # --- the forms over the recovered basis ---------------------------------------------------
    def _pools(self, split):
        return (self.Ztr, len(self.a_tr)) if split == "tr" else (self.Zte, len(self.a_te))

    def logits_A(self, K, split):
        Z, n = self._pools(split)
        if not K:
            return np.zeros((n, self.p))
        idx = list(K)
        return (Z[:, idx] @ np.conj(self.Psi[:, idx]).T).real

    def alpha_K(self, K):
        K = tuple(K)
        if K not in self.alpha_cache:
            idx = list(K)
            self.alpha_cache[K] = np.linalg.lstsq(self.als_G[np.ix_(idx, idx)], self.als_b[idx], rcond=None)[0]
        return self.alpha_cache[K]

    def logits_ALS(self, K, split):
        Z, n = self._pools(split)
        if not K:
            return np.zeros((n, self.p))
        idx = list(K)
        return ((Z[:, idx] * self.alpha_K(K)) @ np.conj(self.Psi[:, idx]).T).real

    def proj_C(self, K):
        Qk = orth(np.concatenate([self.u0[:, None]] + [self.Qs[j] for j in K], 1))
        return Qk @ Qk.T

    def W0_C(self, K):
        Pk = self.proj_C(K)
        return torch.tensor(np.concatenate([self.Wa @ Pk, self.Wb @ Pk], 1), dtype=torch.float32)

    def agree(self, form, K, split, target):
        if form == "A":
            return self._agree_logits(self.logits_A(K, split), target)
        if form == "ALS":
            return self._agree_logits(self.logits_ALS(K, split), target)
        if form == "B":
            return self._agree_logits(self.logits_B(K, split), target)
        if form == "C":
            return self._agree_logits(self.logits_C(K, split), target)
        raise ValueError(form)

    def producer_order(self):
        j = np.arange(self.m)
        return np.lexsort((j, -self.score))          # descending score, ties by candidate index

    def kmatch(self, K):
        km = self.read["k_match"]
        return sorted(set(int(km[j]) for j in K))


# =============================================================================
# FORK NOTICE. `analyze_basis_snapshot` is adapted from mint.py::analyze_snapshot (2026-09-22): the walk,
# random-order controls and cross table are unchanged; candidates are the recovered subspaces (their
# number m varies by snapshot), every kept K is also reported as its DFT-matched frequency set, the
# logit-table energy is dropped, and the operator / certificate / oracle record is added.
# =============================================================================

def analyze_basis_snapshot(snap, rules, n_rand):
    t0 = time.time()
    m = snap.m
    cands = [[j] for j in range(m)]
    prod = snap.producer_order()
    rd = snap.read
    rec = {"net_train_acc_fit": snap.net_train_acc_fit, "net_train_acc_true": snap.net_train_acc_true,
           "net_test_acc": snap.net_test_acc,
           "e_hat": snap.e_hat, "id_score_max": float(snap.id_score.max()),
           "id_score_0": float(snap.id_score[0]), "ops": snap.ops,
           "read": rd,
           "score": [float(x) for x in snap.score], "support": [int(x) for x in snap.support],
           "score_dft": [float(x) for x in snap.score_dft],
           "dc_energy_rec": snap.dc_energy_rec, "dc_energy_dft": float(snap.rd["dc"].sum()),
           "prod_order": [int(i) for i in prod], "dom": [int(x) for x in snap.dom],
           "dom_dft": [int(x) for x in snap.dom_dft],
           "unit_conc_median": float(np.median(snap.E.max(1) / (snap.E.sum(1) + 1e-300))) if m else None,
           "walks": {}, "cross": {}}
    allK = tuple(range(m))
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
                traj.append([int(n_aud), len(K), snap.aud(form)(kept), snap.heldout(form, K), snap.units(K)])

            r = census_walk(snap.aud(form), [], cands, prod, tol=tol, on_change=_on_change)
            K = _flat(r["kept"])
            out = {"K": list(K), "k": len(K), "K_match": snap.kmatch(K),
                   "e_base": r["e_base_gate"], "e_final": r["e_gate_final"], "n_aud": r["n_auditions"],
                   "admit_order": [int(ci) for ci in r["admitted"]],
                   "heldout": snap.heldout(form, K), "train_label": snap.train_label_acc(form, K),
                   "units": snap.units(K), "traj": traj}
            rand = []
            for ri in range(n_rand):
                oi = np.random.RandomState(RAND_SEED0 + ri).permutation(m)
                rr = census_walk(snap.aud(form), [], cands, oi, tol=tol)
                KR = _flat(rr["kept"])
                rand.append({"K_match": snap.kmatch(KR), "k": len(KR), "e_final": rr["e_gate_final"],
                             "heldout": snap.heldout(form, KR), "units": snap.units(KR)})
            out["rand"] = rand
            rec["walks"][form][rname] = out
    for rname in rules:
        rec["cross"][rname] = {}
        for gform in FORMS:
            K = tuple(rec["walks"][gform][rname]["K"])
            rec["cross"][rname][gform] = {f: snap.heldout(f, K) for f in FORMS}
    rec["seconds"] = time.time() - t0
    return rec

# ============================ end of adapted fork ============================


# =============================================================================
# Instrument checks on banked nets
# =============================================================================

def _walk_K(snap, form, tol):
    r = census_walk(snap.aud(form), [], [[j] for j in range(snap.m)], snap.producer_order(), tol=tol)
    return _flat(r["kept"])


def instrument_checks(mint_tag="m1", labels=("e39999", "e14000")):
    """I-1: the exact DFT basis injected here reproduces m1's recorded K and held-out (all forms, rules).
    I-2: at the recovered basis, form B at K = all is bit-exact and form C at K = all matches to float error.
    I-3: ALS joint closed form == explicit lstsq with per-row intercepts on a train subsample."""
    split = split_pairs()
    rules = gate_rules()
    with open(os.path.join(DATA_DIR, mint_tag, "mint.json")) as f:
        m1 = json.load(f)["records"]
    all_labels, epochs, states = _load_snaps(os.path.join(_run_dir(mint_tag, "true"), "snapshots.npz"))
    out = {}
    for lab in labels:
        st = states[all_labels.index(lab)]
        sd = BasisSnapshot(st, split, basis="dft")
        r1 = m1[f"true:{lab}"]
        res = {"e_hat": sd.e_hat,
               "score_maxrel": float(np.max(np.abs(sd.score - np.asarray(r1["score"])) /
                                            (np.asarray(r1["score"]) + 1e-300))),
               "dom_equal": bool(np.array_equal(sd.dom + 1, np.asarray(r1["dom"])))}
        match, dh = {}, 0.0
        for form in FORMS:
            for rn, tol in rules.items():
                K = _walk_K(sd, form, tol)
                Kw = [j + 1 for j in K]
                h = sd.heldout(form, K)
                match[f"{form}/{rn}"] = bool(Kw == r1["walks"][form][rn]["K"])
                dh = max(dh, abs(h - r1["walks"][form][rn]["heldout"]))
        res["K_equal"] = match
        res["K_equal_all"] = all(match.values())
        res["heldout_maxdiff"] = dh
        # I-2 / I-3 at the recovered basis
        sr = BasisSnapshot(st, split)
        allK = tuple(range(sr.m))
        res["rec_m"] = sr.m
        res["rec_B_all_bitexact"] = bool(torch.equal(sr.logits_B(allK, "tr"), sr.L_tr))
        res["rec_C_all_maxabs"] = float((sr.logits_C(allK, "tr") - sr.L_tr).abs().max())
        rng = np.random.RandomState(0)
        K = tuple(int(x) for x in rng.choice(sr.m, 3, replace=False))
        idx = list(K)
        sub = rng.choice(len(sr.a_tr), 300, replace=False)
        L = sr.L_tr.double().numpy()[sub]
        X = np.stack([(sr.Ztr[sub, j][:, None] * np.conj(sr.Psi[:, j])[None, :]).real.ravel() for j in K], 1)
        Xi = np.zeros((X.shape[0], len(sub)))
        Xi[np.arange(X.shape[0]), np.repeat(np.arange(len(sub)), P)] = 1.0
        coef = np.linalg.lstsq(np.concatenate([X, Xi], 1), L.ravel(), rcond=None)[0][:len(K)]
        Lc = L - L.mean(1, keepdims=True)                      # the closed form, same subsample
        Pc = sr.Psi - sr.Psi.mean(0, keepdims=True)
        Ar, Ai, Br, Bi = sr.Ztr[sub].real, sr.Ztr[sub].imag, Pc.real, Pc.imag
        bb = (Ar * (Lc @ Br)).sum(0) + (Ai * (Lc @ Bi)).sum(0)
        GG = ((Ar.T @ Ar) * (Br.T @ Br) + (Ar.T @ Ai) * (Br.T @ Bi)
              + (Ai.T @ Ar) * (Bi.T @ Br) + (Ai.T @ Ai) * (Bi.T @ Bi))
        closed = np.linalg.lstsq(GG[np.ix_(idx, idx)], bb[idx], rcond=None)[0]
        res["I3_als_maxdiff"] = float(np.abs(closed - coef).max())
        out[lab] = res
    # I-2 at an untrained init as well (seed 0)
    torch.manual_seed(0)
    net = GrokMLP(p=P, hidden_dims=HIDDEN)
    st = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
    sr = BasisSnapshot(st, split)
    allK = tuple(range(sr.m))
    out["rinit_s0"] = {"rec_m": sr.m, "n_pairs": sr.read["n_pairs"], "n_real": sr.read["n_real"],
                       "rec_B_all_bitexact": bool(torch.equal(sr.logits_B(allK, "tr"), sr.L_tr)),
                       "rec_C_all_maxabs": float((sr.logits_C(allK, "tr") - sr.L_tr).abs().max())}
    return out


# =============================================================================
# Modal functions (names prefixed `basis_`: mint.py's functions share the app)
# =============================================================================

@app.function(cpu=4.0, memory=2048, timeout=1800, volumes={DATA_DIR: volume})
def basis_gates(mint_tag: str = "m1"):
    volume.reload()
    t0 = time.time()
    res = {"designed": designed_checks(), "instrument": instrument_checks(mint_tag), "gens": list(GENS)}
    for k, v in res["designed"].items():
        print(f"  {k}: {v}", flush=True)
    for k, v in res["instrument"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"gates {time.time() - t0:.0f}s rss={peak_rss_mb():.0f}MB", flush=True)
    return res


def _plan(mint_tag, rand_every, smoke):
    """[(run, label, epoch, with_rand)] in the order they are analysed."""
    jobs = []
    for run in ("true", "shuffled"):
        labels, epochs, _ = _load_snaps(os.path.join(_run_dir(mint_tag, run), "snapshots.npz"))
        order = np.argsort(epochs, kind="stable")
        labels = [labels[i] for i in order]
        epochs = [epochs[i] for i in order]
        last = max(epochs)
        for lab, e in zip(labels, epochs):
            rnd = e == -1 or e == last or (e + 1) % rand_every == 0 or e % rand_every == 0
            jobs.append((run, lab, e, bool(rnd)))
    jobs += [("init", f"init_s{s}", -1, True) for s in INIT_SEEDS]
    if smoke:
        keep = {("true", "init"), ("true", "e5000"), ("true", "e12000"), ("true", "e14000"),
                ("true", "e39999"), ("shuffled", "e39999"), ("init", "init_s0")}
        jobs = [j for j in jobs if (j[0], j[1]) in keep]
    return jobs


@app.function(cpu=4.0, memory=2048, timeout=4 * 3600, volumes={DATA_DIR: volume})
def basis_run(tag: str = "b1", mint_tag: str = "m1", smoke: int = 0, n_rand: int = N_RAND,
              rand_every: int = 1000, resume: int = 1, ckpt_every: int = 20):
    """One container: every snapshot of `mint_tag` (true, shuffled, untrained inits) through steps 1-5.
    Checkpoints every `ckpt_every` records to /data/<tag>/basis_partial.json; resumes from it."""
    volume.reload()
    t0 = time.time()
    split = split_pairs()
    rules = gate_rules()
    d = os.path.join(DATA_DIR, tag)
    os.makedirs(d, exist_ok=True)
    part = os.path.join(d, "basis_partial.json")
    recs = {}
    if resume and os.path.exists(part):
        with open(part) as f:
            recs = json.load(f)
        print(f"resumed {len(recs)} records", flush=True)
    gates = {"designed": designed_checks(), "instrument": instrument_checks(mint_tag)}
    print(f"gates done at {time.time() - t0:.0f}s: D1 dft_ov_min={gates['designed']['D1_shift']['dft_ov_min']:.12f} "
          f"I1 K_equal={[v.get('K_equal_all') for v in gates['instrument'].values()]}", flush=True)
    jobs = _plan(mint_tag, rand_every, smoke)
    cache = {}
    n_new = 0
    for run, lab, ep, rnd in jobs:
        key = f"{run}:{lab}"
        if key in recs:
            continue
        if run == "init":
            s = int(lab.split("_s")[1])
            torch.manual_seed(s)
            net = GrokMLP(p=P, hidden_dims=HIDDEN)
            state = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
            snap = BasisSnapshot(state, split)
        else:
            if run not in cache:
                labels, epochs, states = _load_snaps(os.path.join(_run_dir(mint_tag, run), "snapshots.npz"))
                tl = np.load(os.path.join(_run_dir(mint_tag, run), "train_labels.npy"))
                cache = {run: (labels, states, tl)}                 # one run's states in memory at a time
            labels, states, tl = cache[run]
            snap = BasisSnapshot(states[labels.index(lab)], split, train_labels=tl)
        rec = analyze_basis_snapshot(snap, rules, n_rand if rnd else 0)
        rec.update({"label": lab, "epoch": ep, "run": run})
        recs[key] = rec
        n_new += 1
        rd = rec["read"]
        print(f"  [{run}] {lab:>7s} test={rec['net_test_acc']:.3f} e_hat={rec['e_hat']} "
              f"bij={sum(o['bijection'] for o in rec['ops'].values())}/{len(GENS)} m={rd['m']} "
              f"real={rd['n_real']} xg={rd['xg_mean']:.4f}/{rd['xg_min']:.4f} "
              f"dft={rd['dft_ov_mean']:.4f}/{rd['dft_ov_min']:.4f} ph={rd['phase_err_max']:.2e} "
              f"kB={rec['walks']['B']['lt1']['k']} kC={rec['walks']['C']['lt1']['k']} "
              f"hB={rec['walks']['B']['lt1']['heldout']:.3f} hC={rec['walks']['C']['lt1']['heldout']:.3f} "
              f"{rec['seconds']:.1f}s", flush=True)
        if n_new % ckpt_every == 0:
            with open(part, "w") as f:
                json.dump(recs, f)
            volume.commit()
    out = {"tag": tag, "mint_tag": mint_tag, "p": P, "gens": list(GENS), "g_ref": GENS[0],
           "rules": rules, "forms": list(FORMS), "n_rand": n_rand, "rand_every": rand_every,
           "smoke": smoke, "gates": gates, "records": recs,
           "seconds": time.time() - t0, "peak_rss_mb": peak_rss_mb()}
    path = os.path.join(d, "basis.json")
    with open(path, "w") as f:
        json.dump(out, f)
    volume.commit()
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB), {len(recs)} records, "
          f"{time.time() - t0:.0f}s, rss={peak_rss_mb():.0f}MB", flush=True)
    return path
