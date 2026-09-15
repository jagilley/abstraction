"""Part 2 stimuli: grammar violations and legal surprises, labelled by the exact oracle.

Model-independent, built once per grammar and reused by every checkpoint.

Each stimulus is a flat window (T+1 tokens at uniform random phase over a stream of two
i.i.d. sequences, generated WITH latents) and an edit of one constituent: the node u at
tree level L-j spanning 2^j leaves, aligned so it starts at window index e.

  swap   u's feature is replaced by a different one and its subtree regrown from the
         grammar. The regrown constituent is internally legal; whether the stream is
         still legal is NOT assumed -- the oracle decides.
  rare   u keeps its feature but switches to its lowest-weight synonym (j >= 1), subtree
         regrown below. Legal by construction; the oracle is asserted to agree.
  none   no edit (natural controls).

Labels, all from `flat_predictive` on the edited window:
  t_v    first window index whose token has EXACT probability 0 under the true DGP given
         the window so far (the moment a noise-free Bayesian knows the rules were broken;
         -1 if none inside the window).
  k_star the smallest observer k with p_k(token at t_v) = 0: how many grammar levels you
         must know to be offended. Supports are nested (supp p_{k+1} within supp p_k), so
         every observer above k_star also assigns 0. k_star >= j+1 by construction.
Also stored: p_L on the edited and original windows, and the eps-noise observer on both
(the reference that stays defined after a violation).

Run:
  modal run -m rhm.logit_reading.stimuli::build_stimuli --n 256 --tag smoke
  modal run --detach -m rhm.logit_reading.stimuli::build_stimuli --n 16384 --tag a1
  # phasic-test set: swap edits only, no reference observers
  modal run --detach -m rhm.logit_reading.stimuli::build_stimuli --n 65536 --seed 2027 \
      --swap-only --no-with-reference --tag swap65k
"""

import json
import os
import time

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, tb_key


def make_edits(rules, rule_w, n, seed, e_lo=16, e_hi=36, j_max=4, frac=(0.5, 0.25, 0.25)):
    """Numpy only. Returns dict of windows (orig/edit) and edit metadata."""
    from rhm.logit_reading.grammar import generate, realise
    L = len(rules)
    v, m, s = rules[0].shape
    T = s ** L
    T1 = T + 1
    rng = np.random.default_rng(seed)
    leaves, feats, rcs = generate(rules, rule_w, 2 * n, rng, return_latents=True)
    stream = np.concatenate([leaves[0::2], leaves[1::2]], axis=1)          # (n, 2T)
    phase = rng.integers(0, T, size=n)
    etype = rng.choice(3, size=n, p=list(frac))                             # 0 swap 1 rare 2 none
    j = np.where(etype == 1, rng.integers(1, j_max + 1, size=n), rng.integers(0, j_max + 1, size=n))
    e = np.zeros(n, dtype=np.int64)
    for w in range(n):
        cands = [x for x in range(e_lo, e_hi + 1) if (phase[w] + x) % (s ** j[w]) == 0]
        e[w] = rng.choice(cands)
    edit = stream.copy()
    for w in range(n):
        if etype[w] == 2:
            continue
        pos = phase[w] + e[w]
        which, i = divmod(pos, T)
        row = 2 * w + which
        d = L - j[w]
        u = i >> j[w]
        f = feats[d][row, u]
        if etype[w] == 0:
            f2 = rng.choice([x for x in range(v) if x != f])
            new = realise(rules, rule_w, np.array([f2]), d, rng)[0]
        else:
            r0 = rcs[d][row, u]
            wts = (np.full(m, 1.0 / m) if rule_w is None else rule_w[d][f]).copy()
            wts[r0] = np.inf
            r_new = int(np.argmin(wts)) if rule_w is not None else \
                int(rng.choice([x for x in range(m) if x != r0]))
            kids = rules[d][f, r_new]
            new = np.concatenate([realise(rules, rule_w, np.array([kd]), d + 1, rng)[0]
                                  for kd in kids])
        edit[w, pos:pos + s ** j[w]] = new
    idx = phase[:, None] + np.arange(T1)[None, :]
    w_orig = np.take_along_axis(stream, idx, 1)
    w_edit = np.take_along_axis(edit, idx, 1)
    diff = w_orig != w_edit
    first_diff = np.where(diff.any(1), diff.argmax(1), -1)
    return dict(windows_orig=w_orig, windows_edit=w_edit, phase=phase, etype=etype, j=j, e=e,
                first_diff=first_diff, region_end=e + s ** j)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=65536)
def build_stimuli(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
                  alpha: float = 1.0, weight_seed: int = 1, n: int = 16384, seed: int = 2026,
                  noise_eps: float = 0.01, chunk: int = 8, tag: str = "a1",
                  swap_only: bool = False, with_reference: bool = True):
    """swap_only: every window is a swap edit (the phasic test uses nothing else).
    with_reference: also compute p_L on the original windows and the eps-noise observer on
    both (Part 2 persistence); off for the phasic stimulus sets, which only need p_L on the
    edited windows and the observer ladder."""
    import torch
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    from rhm.logit_reading.flat_oracle import flat_predictive

    L = depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)
    t0 = time.time()
    st = make_edits(rules, rule_w, n, seed, **({"frac": (1.0, 0.0, 0.0)} if swap_only else {}))
    print(f"edits built {time.time() - t0:.1f}s  types {np.bincount(st['etype'])}", flush=True)
    We, Wo = st["windows_edit"], st["windows_orig"]
    T1 = We.shape[1]

    ptok = np.zeros((n, T1, L + 1))
    for k in range(L + 1):
        t0 = time.time()
        ck = chunk if k == L else max(chunk, 64)
        pr = flat_predictive(We, rules, k, device="cuda", chunk=ck, rule_w=rule_w)
        if k == L:
            pL_edit = pr.numpy()
        ptok[:, :, k] = np.take_along_axis(pr.numpy(), We[..., None], -1)[..., 0]
        print(f"  edit observer k={k} {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    if with_reference:
        pL_orig = flat_predictive(Wo, rules, L, device="cuda", chunk=chunk, rule_w=rule_w).numpy()
        pE_edit = flat_predictive(We, rules, L, device="cuda", chunk=chunk, rule_w=rule_w,
                                  noise_eps=noise_eps).numpy()
        pE_orig = flat_predictive(Wo, rules, L, device="cuda", chunk=chunk, rule_w=rule_w,
                                  noise_eps=noise_eps).numpy()
        print(f"  orig + noise observers {time.time() - t0:.1f}s", flush=True)

    ptok = np.nan_to_num(ptok, nan=0.0)
    zero = ptok[:, 1:, L] == 0.0
    t_v = np.where(zero.any(1), zero.argmax(1) + 1, -1)
    k_star = np.full(n, -1)
    for w in np.where(t_v > 0)[0]:
        k_star[w] = int(np.argmax(ptok[w, t_v[w]] == 0.0))
        # nested supports: every observer above k_star must also exclude the token
        assert (ptok[w, t_v[w], k_star[w]:] == 0.0).all(), (w, ptok[w, t_v[w]])
    legal = st["etype"] != 0
    assert (t_v[legal] == -1).all(), "a legal stream hit probability 0 -- oracle/generator bug"
    if with_reference:
        ok_orig = np.take_along_axis(pL_orig, Wo[..., None], -1)[..., 0][:, 1:]
        assert (ok_orig > 0).all(), "an unedited window hit probability 0"
    assert (t_v[t_v > 0] >= st["e"][t_v > 0]).all()
    assert (k_star[t_v > 0] >= st["j"][t_v > 0] + 1).all()

    sw = st["etype"] == 0
    summary = {
        "n": n, "noise_eps": noise_eps, "alpha": alpha,
        "types": np.bincount(st["etype"], minlength=3).tolist(),
        "swap_violation_rate": float((t_v[sw] > 0).mean()),
        "k_star_hist": np.bincount(k_star[k_star >= 0], minlength=L + 1).tolist(),
        "k_star_by_j": {int(jj): np.bincount(k_star[sw & (st["j"] == jj) & (k_star >= 0)],
                                             minlength=L + 1).tolist() for jj in range(5)},
        "delay_tv_minus_e_by_kstar": {int(k): float((t_v - st["e"])[k_star == k].mean())
                                      for k in range(L + 1) if (k_star == k).any()},
        "min_legal_prob": float(ok_orig.min()) if with_reference else None,
        "swap_only": swap_only, "with_reference": with_reference,
    }
    print(json.dumps(summary, indent=1), flush=True)

    key = tb_key(v, s, L, m)
    out_dir = f"{DATA_DIR}/{key}/logit_reading"
    os.makedirs(out_dir, exist_ok=True)
    ref = (dict(pL_orig=pL_orig.astype(np.float32), pE_edit=pE_edit.astype(np.float32),
                pE_orig=pE_orig.astype(np.float32)) if with_reference else {})
    np.savez_compressed(f"{out_dir}/stimuli_{tag}.npz", **st, t_v=t_v, k_star=k_star,
                        ptok=ptok.astype(np.float32), pL_edit=np.nan_to_num(pL_edit).astype(np.float32),
                        **ref)
    with open(f"{out_dir}/stimuli_{tag}.json", "w") as f:
        json.dump({"summary": summary, "config": dict(v=v, s=s, L=L, m=m, rule_seed=rule_seed,
                   alpha=alpha, weight_seed=weight_seed, n=n, seed=seed)}, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"saved -> {out_dir}/stimuli_{tag}.npz", flush=True)
    return summary
