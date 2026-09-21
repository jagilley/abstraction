"""The generating parse of the Part 2 stimuli, original and edited, per level.

`stimuli.make_edits` stores the token windows but not the latents of the EDITED stream,
so "what is the level-l feature of the constituent containing window index t" has no
stored answer for an edited window. This module replays `make_edits`'s RNG exactly --
same seed, same call order, with a latent-recording `realise` -- and asserts the
reconstructed windows, phases and edit metadata are BIT-IDENTICAL to the cached
`stimuli_<tag>.npz` before returning anything. (Same pattern as `coeruleus/events.py`
reconstructing the corruption mask from `dual_ladder`'s RNG.)

Output, cached as `parse_<tag>.npz`:

    y_orig[l-1, w, t]   level-l feature of the constituent containing window index t,
    y_edit[l-1, w, t]   in the generating tree of the original / edited stream, l = 1..6

The edited stream's generating tree is the original tree with the swapped node's feature
replaced and its subtree replaced by the regrowth; every ancestor keeps its feature (that
is what makes the stream illegal). So `y_edit != y_orig` exactly on the nodes strictly
inside the edited span, which is the model-independent CONSEQUENCE label this node needs:
a violation is consequential for a (position, level) query iff the query's true answer
differs between the original and the edited window.

Run:
  modal run -m rhm.logit_reading.striatum.parse::build_parse --tag a1
  modal run -m rhm.logit_reading.striatum.parse::build_parse --tag swap65k \
      --n 65536 --seed 2027 --swap-only
"""

import json
import os

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, tb_key


def realise_lat(rules, rule_w, feat, d, rng):
    """`grammar.realise` with the latents kept. Consumes the RNG identically."""
    L = len(rules)
    cur = np.asarray(feat)[:, None]
    lat = {d: cur}
    for dd in range(d, L):
        nn_ = cur.shape[1]
        if rule_w is None:
            rc = rng.integers(0, m_of(rules), size=cur.shape)
        else:
            cw = np.cumsum(rule_w[dd], 1)
            rc = np.minimum((rng.random(cur.shape)[..., None] >= cw[cur]).sum(-1),
                            rules[0].shape[1] - 1)
        cur = rules[dd][cur, rc].reshape(cur.shape[0], nn_ * rules[0].shape[2])
        lat[dd + 1] = cur
    return lat


def m_of(rules):
    return rules[0].shape[1]


def make_edits_parsed(rules, rule_w, n, seed, e_lo=16, e_hi=36, j_max=4,
                      frac=(0.5, 0.25, 0.25)):
    """`stimuli.make_edits`, line for line, plus the edited stream's latents.

    Returns the same dict, with `feats_orig` / `feats_edit` (lists of (2n, s^d) arrays).
    """
    from rhm.logit_reading.grammar import generate
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
    feats_edit = [f.copy() for f in feats]
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
            lat = realise_lat(rules, rule_w, np.array([f2]), d, rng)
            new = lat[L][0]
            for dd in range(d, L + 1):
                w_ = s ** (dd - d)
                feats_edit[dd][row, u * w_:(u + 1) * w_] = lat[dd][0]
        else:
            r0 = rcs[d][row, u]
            wts = (np.full(m, 1.0 / m) if rule_w is None else rule_w[d][f]).copy()
            wts[r0] = np.inf
            r_new = int(np.argmin(wts)) if rule_w is not None else \
                int(rng.choice([x for x in range(m) if x != r0]))
            kids = rules[d][f, r_new]
            lats = [realise_lat(rules, rule_w, np.array([kd]), d + 1, rng) for kd in kids]
            new = np.concatenate([lt[L][0] for lt in lats])
            for dd in range(d + 1, L + 1):
                w_ = s ** (dd - d - 1)
                for ki, lt in enumerate(lats):
                    feats_edit[dd][row, (u * s + ki) * w_:(u * s + ki + 1) * w_] = lt[dd][0]
        edit[w, pos:pos + s ** j[w]] = new
    idx = phase[:, None] + np.arange(T1)[None, :]
    w_orig = np.take_along_axis(stream, idx, 1)
    w_edit = np.take_along_axis(edit, idx, 1)
    diff = w_orig != w_edit
    first_diff = np.where(diff.any(1), diff.argmax(1), -1)
    return dict(windows_orig=w_orig, windows_edit=w_edit, phase=phase, etype=etype, j=j, e=e,
                first_diff=first_diff, region_end=e + s ** j,
                feats_orig=feats, feats_edit=feats_edit)


def level_answers(feats, phase, T1, L, s, two_rows=True):
    """y[l-1, w, t] = feature of the level-l constituent (span s^l) containing window
    index t, for l = 1..L. `feats[d]` is (rows, s^d) with row = 2w + which when
    `two_rows` (the stimulus stream is two concatenated sequences per window)."""
    n = len(phase)
    T = s ** L
    t = np.arange(T1)[None, :]
    p = phase[:, None] + t                                   # (n, T1) stream coord
    which, i = np.divmod(p, T)
    row = (2 * np.arange(n)[:, None] + which) if two_rows else np.broadcast_to(
        np.arange(n)[:, None], p.shape)
    out = np.zeros((L, n, T1), dtype=np.int16)
    for l in range(1, L + 1):
        d = L - l
        node = i >> l
        if two_rows:
            out[l - 1] = feats[d][row, node]
        else:
            out[l - 1] = feats[d].reshape(n, -1, s ** d)[
                np.arange(n)[:, None], which, node]
    return out


def clean_answers(fs, phase, T1, L, s, n_seq):
    """Same, for `altitude.units.windows_with_parse` output (fs[d] is (n, n_seq*s^d))."""
    n = len(phase)
    T = s ** L
    p = phase[:, None] + np.arange(T1)[None, :]
    which, i = np.divmod(p, T)
    out = np.zeros((L, n, T1), dtype=np.int16)
    rows = np.arange(n)[:, None]
    for l in range(1, L + 1):
        d = L - l
        out[l - 1] = fs[d][rows, which * (s ** d) + (i >> l)]
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384, cpu=2.0)
def build_parse(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
                alpha: float = 1.0, weight_seed: int = 1, n: int = 16384, seed: int = 2026,
                tag: str = "a1", swap_only: bool = False):
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    volume.reload()
    L = depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)
    st = make_edits_parsed(rules, rule_w, n, seed,
                           **({"frac": (1.0, 0.0, 0.0)} if swap_only else {}))

    key = tb_key(v, s, L, m)
    out_dir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{out_dir}/stimuli_{tag}.npz")
    for k in ("windows_orig", "windows_edit", "phase", "etype", "j", "e",
              "first_diff", "region_end"):
        assert (S[k] == st[k]).all(), f"replay mismatch on {k} -- RNG order differs"
    print("replay bit-identical to stimuli_%s.npz on 8 arrays" % tag, flush=True)

    T1 = st["windows_edit"].shape[1]
    y_orig = level_answers(st["feats_orig"], st["phase"], T1, L, s)
    y_edit = level_answers(st["feats_edit"], st["phase"], T1, L, s)
    # the leaves must reproduce the windows exactly (level-0 sanity on both trees)
    assert (y_orig[:, st["etype"] == 2] == y_edit[:, st["etype"] == 2]).all(), "none-window edited"

    cons = (y_orig != y_edit)
    summary = {
        "n": n, "tag": tag,
        "cons_rate_by_level": [float(cons[l].mean()) for l in range(L)],
        "cons_rate_by_level_swap": [float(cons[l][st["etype"] == 0].mean()) for l in range(L)],
        "cons_rate_by_level_rare": [float(cons[l][st["etype"] == 1].mean()) for l in range(L)]
        if (st["etype"] == 1).any() else None,
    }
    print(json.dumps(summary, indent=1), flush=True)
    np.savez_compressed(f"{out_dir}/parse_{tag}.npz", y_orig=y_orig.astype(np.int16),
                        y_edit=y_edit.astype(np.int16))
    with open(f"{out_dir}/parse_{tag}.json", "w") as f:
        json.dump(summary, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"saved -> {out_dir}/parse_{tag}.npz", flush=True)
    return summary
