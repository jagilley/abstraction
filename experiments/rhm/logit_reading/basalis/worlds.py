"""The event populations: two worlds that are the same event at the flagged token and
diverge only in the continuation, plus the background a false alarm lands on.

The brief's structural fact is that at the violating token itself a glitch and a structural
edit are indistinguishable -- both are an impossible token given the prefix -- and the
information that tells them apart arrives over the following tokens. To measure that without
the venue doing the work, the two worlds here are built from the SAME stream, at the SAME
position, from the same `swap65k` stimulus:

  edit      `windows_edit[w]`, flagged at `t_v`: one constituent at tree level L-j was
            swapped and regrown LEGALLY under a new feature, so the prefix carries a legal
            but different subtree and the continuation is the original stream, which is what
            makes `t_v` impossible. Note j = 0 is a one-leaf swap -- a single-token
            substitution with no subtree at all, i.e. a glitch drawn from the grammar's own
            alphabet -- so `edit` is reported split at j >= 1 (structural) and j = 0 (point).
  glitch_m  `windows_orig[w]` -- the same window with no edit -- with the token at the same
            `t_v` replaced by a uniform draw. Same stream, same position, same k*/j strata,
            paired one-to-one with the edit event. The prefix is intact, so the continuation
            is consistent with it and the flagged token is the only damage.
  quiet     `windows_orig` of DISJOINT windows at a random position, nothing flagged: what a
            false alarm costs. The consumer here is event-triggered (the detector's job is
            bounded already -- `coeruleus/` Q2 -- and is not re-asked), so the base rate
            enters as the rate of quiet triggers in the mixed world.

`glitch_nat` is the other venue: `coeruleus/events.py`'s isolated corruptions on the cached
eps-corrupted windows, where the exact eps-observer ladder is on file and the family ceiling
of `coeruleus/` Q1 applies, so the prize is comparable to that round's numbers.
"""

import numpy as np


def load_swap_stimuli(stim_tag, key="v16_s2_L6_m4_distinct", data_dir="/data", full=False):
    """Only the small arrays unless `full` (pL_edit alone is ~270 MB)."""
    z = np.load(f"{data_dir}/{key}/logit_reading/stimuli_{stim_tag}.npz")
    keys = (z.files if full else
            [k for k in ("windows_edit", "windows_orig", "t_v", "k_star", "j", "e", "etype",
                         "first_diff", "region_end", "phase") if k in z.files])
    return {k: z[k] for k in keys}


def eligible(S, H, t_lo=16):
    """Swap windows whose violation is far enough inside the window for offsets 0..H."""
    T = S["windows_edit"].shape[1] - 1
    tv = S["t_v"]
    return np.where((S["etype"] == 0) & (tv >= t_lo) & (tv <= T - 1 - H))[0]


def build_arms(S, H, n_events, seed=0, t_lo=16, v=16):
    """(arms, meta). Every arm is dict(win, pos, label) with `win` the (n, T+1) token
    windows it runs on and `pos` the flagged position."""
    rng = np.random.default_rng(seed)
    T = S["windows_edit"].shape[1] - 1
    elig = eligible(S, H, t_lo)
    rng.shuffle(elig)
    n_ev = min(n_events, len(elig) // 2)
    ev, qu = np.sort(elig[:n_ev]), np.sort(elig[n_ev:2 * n_ev])
    tv = S["t_v"][ev]

    # glitch twin: the same original stream, the same position, a uniform draw that lands
    # on a different token (the effective-corruption convention of `coeruleus/events.py`).
    Wg = np.array(S["windows_orig"][ev])
    ar = np.arange(len(ev))
    orig_tok = Wg[ar, tv].copy()
    draw = rng.integers(0, v, len(ev))
    while True:
        bad = draw == orig_tok
        if not bad.any():
            break
        draw[bad] = rng.integers(0, v, int(bad.sum()))
    Wg[ar, tv] = draw

    # quiet: disjoint windows, unedited, a random position in the same range as t_v
    Wq = np.array(S["windows_orig"][qu])
    tq = rng.integers(t_lo, T - H, len(qu))

    arms = {
        "edit": {"win": np.array(S["windows_edit"][ev]), "pos": tv, "src": ev},
        "glitch_m": {"win": Wg, "pos": tv.copy(), "src": ev},
        "quiet": {"win": Wq, "pos": tq, "src": qu},
    }
    meta = {"n_eligible": int(len(elig)), "n_events": int(n_ev),
            "kstar": S["k_star"][ev], "j": S["j"][ev], "e": S["e"][ev],
            "tv": tv, "delay": (tv - S["e"][ev]),
            "orig_token": orig_tok, "glitch_token": draw,
            "edit_token": S["windows_edit"][ev][ar, tv],
            "tv_beyond_region": (tv >= S["region_end"][ev]).astype(bool)}
    return arms, meta


def natural_glitch_arm(rules, rule_w, eps_data, H, n_windows=4096, eval_seed=4242, c_lo=16):
    """`coeruleus/events.py`'s isolated corruptions on the cached ladder windows."""
    from rhm.logit_reading.coeruleus.events import assert_cache_matches, events_from_mask
    w, w_clean, corrupt, phase = assert_cache_matches(rules, rule_w, eps_data, n_windows,
                                                      eval_seed)
    T = w.shape[1] - 1
    win, c = events_from_mask(corrupt, H, c_lo=c_lo, isolate=True)
    ok = c <= T - 1 - H
    win, c = win[ok], c[ok]
    return {"win": w[win], "pos": c, "src": win}, {"windows": w, "clean": w_clean,
                                                   "corrupt": corrupt, "win_idx": win,
                                                   "c": c, "n_events": int(len(c))}
