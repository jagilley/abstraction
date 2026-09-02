"""trap_menu.py — THE TRAP CHANNEL: worthless answers, installed in the world's own menu.

Out of the Modal app on purpose (`../questions.py`'s convention, itself
`../../conductor/policy.py`'s): every rule here is auditable and gate-able with no GPU, and
`antiphon.py` calls exactly one function per cycle.

WHAT A DISTRACTOR IS (`DESIGN.md` section 1, fact F4). Every true level-L flat's two halves are
true level-(L-1) flats — 0 exceptions at L3/L4/L5 on the DGP's own tables. So a key whose half is
not a true row is junk WITH CERTAINTY, and `Miner.build`'s ratchet refuses it forever. The half
the QUESTION fixes is the one the era's damage cell does not cover, and it reaches the miner as
the question set it (confirmed on the banked `an_s0` logs: every at-support key's clean half is a
legal parse of the clean node, in every arm, at both eras). Therefore:

    a DISTRACTOR is a menu candidate whose CLEAN HALF parses to a key that is not a row of the
    true table at that level.

Its answer cannot contribute a true key at the target level, ever. It is legal, solved and mined
at the ordinary rate, drawn from the world's own cell, and d*-orthogonal by construction (its
defining property is a parse of the CLEAN blocks; d* is a property of the DAMAGE — measured at
TV 0.074 / 0.039 in `phase0_trap.py` [3]).

WHY NOT A FOREIGN GRAMMAR. `crystallize.units.apply_move` materialises every span through a
max-sum DP over THIS world's `rules_t` and renders it through `canon`, so the agent cannot produce
a foreign-legal answer; and `run_arm` block (b) mines only what the world's exact grader passed.
A foreign channel is therefore never mined and degenerates into "burn budget, bank nothing".
Rejected in `DESIGN.md` section 1 (F3), with reasons.

THE THREE THINGS THE TRAP MUST NOT MOVE, and how each is held:

  * VOLUME      distractors SUBSTITUTE into menu slots; K in, K out. Asserted here.
  * DIFFICULTY  the substitution is over slots chosen uniformly at random across the WHOLE menu,
                the head's `n_pr` included, so the head stays the world's own draw and the d*
                quota is that draw's own histogram. Gate T-3d checks the realised d* histogram
                against the pre-substitution one; the caller logs both every cycle.
  * THE PLANT   nothing here touches the grader, the executor, the value buffer or the
                generator's fine-tune set. The trap is a property of the MENU.

AT f = 0 NOTHING RUNS. `install_trap` returns its inputs unchanged and untouched, which is what
makes the fork bit-identical with the knob off (gate G-T).
"""

import collections

import numpy as np

# names an agent bundle may never carry (merged into `questions._FORBIDDEN` by `containment`)
TRAP_FORBIDDEN = ("trap", "distractor", "is_d", "bank")


# --------------------------------------------------------------------------- #
# the trap atom
# --------------------------------------------------------------------------- #

def clean_level(era_level):
    """The level the clean part of the mined span lives on. The damage cell at level `l` covers
    s**(l-1) blocks of the level-(l+1) span, so the untouched blocks are exactly one level-`l`
    constituent. Below 2 there is no junk to be had: a single block is a level-1 feature and
    every level-1 feature is a valid row."""
    return int(era_level)


def trap_applies(geom, era_level):
    """The trap atom exists only where the mined span has a clean part AND that part is at least
    a level-2 constituent. On `crescendo`'s ladder that is eras 2 and 3 (of 5); era 1's clean
    part is one block, and eras 4-5 have no clean part at all."""
    return bool(geom.get("has_clean")) and clean_level(era_level) >= 2


def is_distractor(clean_keys, truth_flat):
    """Boolean mask: the candidate's clean-half parse is not a row of the true table."""
    return np.array([tuple(int(z) for z in k) not in truth_flat for k in clean_keys], bool)


def clean_keys_of(exact_feats, geom):
    """The exact parse of the CLEAN blocks of the mined span — the trap's defining object, and
    the same array `questions.half_keys` hands the endogenous selectors."""
    if not geom.get("has_clean"):
        return np.zeros((exact_feats.shape[0], 0), dtype=np.int64)
    off = np.asarray(geom["clean_offsets"], dtype=np.int64)
    return exact_feats[:, geom["blk0"] + off]


# --------------------------------------------------------------------------- #
# the per-era distractor bank
# --------------------------------------------------------------------------- #

class DistractorBank:
    """Candidates whose clean half is junk, keyed by that half so the trap can be presented
    UNIFORMLY over the junk-half pool — i.e. maximally as news, which is the presentation a
    novelty judge is most exposed to. Built once per era; ~10 s and ~16 MB per era against an
    ~10.5 s cycle (`phase0_trap.py` [4])."""

    def __init__(self, roots, x, clean, halves, spread="uniform", rng=None):
        self.roots, self.x, self.clean = roots, x, clean
        self.halves = [tuple(int(z) for z in h) for h in halves]
        self.spread = spread
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.by_half = collections.defaultdict(list)
        for i, h in enumerate(self.halves):
            self.by_half[h].append(i)
        self.pool = sorted(self.by_half)
        self._cursor = collections.Counter()

    def __len__(self):
        return len(self.roots)

    def state(self):
        return {"n": len(self.roots), "n_halves": len(self.pool), "spread": self.spread}

    def draw(self, n):
        """`n` row indices. `uniform` cycles the junk-half pool (deterministic round-robin over
        halves, so the presentation is spread rather than frequency-weighted); `native` samples
        rows directly, which reproduces the world's own junk-half frequencies."""
        if not len(self.roots) or n <= 0:
            return np.zeros(0, dtype=np.int64)
        if self.spread == "native":
            return self.rng.integers(0, len(self.roots), size=n).astype(np.int64)
        out = np.empty(n, dtype=np.int64)
        for j in range(n):
            h = self.pool[(self._cursor["h"] + j) % len(self.pool)]
            rows = self.by_half[h]
            out[j] = rows[self._cursor[h] % len(rows)]
            self._cursor[h] += 1
        self._cursor["h"] = (self._cursor["h"] + n) % len(self.pool)
        return out


def build_bank(draw_fn, truth_flat, geom, exact_fn, n_target, *, spread="uniform",
               max_rounds=64, rng=None):
    """Rejection-sample the world's own cell for candidates whose clean half is junk.

    `draw_fn(n, attempt) -> (roots, x, clean)` is `context_instances` with `with_clean=True`;
    `exact_fn(clean) -> (n, n_blocks)` is `macros.exact_features`. Nothing here knows about the
    grammar beyond those two callables, so the bank is testable with synthetic stand-ins.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    R, X, C, H = [], [], [], []
    got = 0
    for attempt in range(max_rounds):
        if got >= n_target:
            break
        want = int(np.ceil((n_target - got) * 1.6)) + 64
        r, x, c = draw_fn(want, attempt)
        ck = clean_keys_of(exact_fn(c), geom)
        m = is_distractor(ck, truth_flat)
        if not m.any():
            continue
        R.append(r[m]); X.append(x[m]); C.append(c[m]); H.append(ck[m])
        got += int(m.sum())
    if not R:
        return None
    return DistractorBank(np.concatenate(R), np.concatenate(X), np.concatenate(C),
                          np.concatenate(H), spread=spread, rng=rng)


# --------------------------------------------------------------------------- #
# the substitution — the port's one call
# --------------------------------------------------------------------------- #

def install_trap(roots, x, clean, *, frac, bank, rng, n_pr):
    """Substitute `round(frac * K)` menu slots, chosen uniformly at random over ALL K slots (the
    head's `n_pr` included), with rows drawn from the bank. Volume is preserved exactly.

    Substituting into the head as well as the tail is what makes the arm that selects nothing
    consume distractors at exactly the menu rate — the null the controls norm asks for — and it
    keeps that arm one object across a dose ladder. At `frac` 0 (or with no bank) this returns
    its inputs untouched, which is the fork's bit-identity with the knob off.
    """
    K = len(roots)
    n_sub = 0 if (not frac or bank is None or len(bank) == 0) else int(round(float(frac) * K))
    if n_sub <= 0:
        return roots, x, clean, np.zeros(K, bool), {"n_sub": 0, "frac": 0.0}
    n_sub = min(n_sub, K)
    slots = rng.choice(K, size=n_sub, replace=False)
    rows = bank.draw(n_sub)
    roots = roots.copy(); x = x.copy(); clean = clean.copy()
    roots[slots] = bank.roots[rows]
    x[slots] = bank.x[rows]
    clean[slots] = bank.clean[rows]
    is_d = np.zeros(K, bool)
    is_d[slots] = True
    return roots, x, clean, is_d, {
        "n_sub": int(n_sub), "frac": float(n_sub) / K,
        "head_n_sub": int(is_d[:n_pr].sum()), "head_frac": float(is_d[:n_pr].mean()),
        "bank": bank.state()}


# --------------------------------------------------------------------------- #
# the trust weights — the guard `phase0_trap.py` [9]/[9b] promotes
# --------------------------------------------------------------------------- #

def trust_weights(lower2_flat, use_counts):
    """The arm's OWN per-entry use record, normalised. `lower2_flat` is the flat of the table one
    level BELOW the clean half — the level where the arm has both a committed table and a beam
    use record — and `use_counts` is `log['entry']['hist']['beam'][that level]`, the priced
    beam's per-entry selection count, which every arm already records and already pays for.

    No truth, no exact map, no clean derivation: this is the arm's own history of which of its
    own entries its own executor actually chose.
    """
    rows = [tuple(int(z) for z in r) for r in lower2_flat]
    u = np.asarray(use_counts, float) if use_counts is not None else np.zeros(len(rows))
    if len(u) != len(rows) or u.sum() <= 0:
        return {r: 1.0 / max(len(rows), 1) for r in rows}
    u = u / u.sum()
    return {r: float(w) for r, w in zip(rows, u)}


# --------------------------------------------------------------------------- #
# gates — all offline, all free
# --------------------------------------------------------------------------- #

def trap_gate(verbose=False):
    """T-3: the menu construction's own suite. Pure arithmetic on synthetic menus, so it runs
    with no grammar, no substrate and no GPU."""
    out = {}
    rng = np.random.default_rng(0)
    K, n_pr, span = 2048, 64, 8
    geom = {"has_clean": True, "blk0": 24, "clean_offsets": [4, 5, 6, 7], "span": span}

    # a synthetic world: 40 "true" clean halves, the rest junk
    all_h = [tuple(int(z) for z in r) for r in rng.integers(0, 8, size=(200, 4))]
    truth_flat = set(all_h[:40])

    def exact_fn(c):
        return c

    def draw_fn(n, attempt):
        idx = rng.integers(0, len(all_h), size=n)
        c = np.zeros((n, 32), np.int64)
        c[:, 28:32] = np.array([all_h[i] for i in idx])
        return rng.integers(0, 8, size=n), rng.integers(0, 8, size=(n, 64)), c

    # T-3a  the atom: `is_distractor` is exactly "clean half is not a true row"
    r0, x0, c0 = draw_fn(512, 0)
    ck = clean_keys_of(exact_fn(c0), geom)
    m = is_distractor(ck, truth_flat)
    out["T-3a_atom"] = bool(all((tuple(int(z) for z in k) in truth_flat) != bool(b)
                                for k, b in zip(ck, m)))

    # T-3b  the bank contains ONLY distractors, and spreads over the junk-half pool
    bank = build_bank(draw_fn, truth_flat, geom, exact_fn, 4096, rng=rng)
    bh = clean_keys_of(exact_fn(bank.clean), geom)
    out["T-3b_bank_pure"] = bool(len(bank) >= 4096
                                 and is_distractor(bh, truth_flat).all())
    d1 = bank.draw(2048)
    spread = len({bank.halves[i] for i in d1})
    out["T-3b_bank_spread"] = bool(spread == len(bank.pool))
    out["T-3b_detail"] = {"n": len(bank), "n_halves": len(bank.pool), "spread_in_2048": spread}

    # T-3c  substitution: volume preserved, the right count, the head consumes at the menu rate
    R, X, C = draw_fn(K, 0)
    ok_frac = {}
    for f in (0.15, 0.5, 0.9):
        r, x, c, is_d, meta = install_trap(R, X, C, frac=f, bank=bank,
                                           rng=np.random.default_rng(7), n_pr=n_pr)
        ok_frac[str(f)] = bool(len(r) == K and len(x) == K and len(c) == K
                               and int(is_d.sum()) == int(round(f * K))
                               and abs(meta["head_frac"] - f) < 0.16)
    out["T-3c_substitution"] = bool(all(ok_frac.values()))
    out["T-3c_detail"] = ok_frac

    # T-3d  at f = 0 NOTHING moves — the fork's bit-identity with the knob off
    r, x, c, is_d, meta = install_trap(R, X, C, frac=0.0, bank=bank,
                                       rng=np.random.default_rng(7), n_pr=n_pr)
    out["T-3d_f0_untouched"] = bool(r is R and x is X and c is C
                                    and not is_d.any() and meta["n_sub"] == 0)

    # T-3e  determinism: same rng seed, same substitution
    a = install_trap(R, X, C, frac=0.5, bank=bank, rng=np.random.default_rng(3), n_pr=n_pr)[3]
    bank._cursor.clear()
    b = install_trap(R, X, C, frac=0.5, bank=bank, rng=np.random.default_rng(3), n_pr=n_pr)[3]
    out["T-3e_deterministic"] = bool(np.array_equal(a, b))

    # T-3f  every substituted slot really carries a distractor
    r, x, c, is_d, _ = install_trap(R, X, C, frac=0.6, bank=bank,
                                    rng=np.random.default_rng(11), n_pr=n_pr)
    ck2 = clean_keys_of(exact_fn(c), geom)
    m2 = is_distractor(ck2, truth_flat)
    out["T-3f_slots_are_distractors"] = bool(m2[is_d].all())

    # T-3g  where the atom does not exist, the trap is off by construction
    out["T-3g_scope"] = bool(trap_applies(geom, 3) and trap_applies(geom, 2)
                             and not trap_applies(geom, 1)
                             and not trap_applies({"has_clean": False}, 4))

    # T-3h  trust weights are a normalised distribution over the arm's own rows, and fall back
    #       to uniform when the arm has no use record yet
    lower = np.array([[0, 1], [2, 3], [4, 5]], np.int64)
    w = trust_weights(lower, [10, 0, 30])
    w0 = trust_weights(lower, None)
    out["T-3h_trust_weights"] = bool(
        abs(sum(w.values()) - 1.0) < 1e-9 and w[(4, 5)] > w[(0, 1)] > w[(2, 3)]
        and abs(sum(w0.values()) - 1.0) < 1e-9 and len(set(w0.values())) == 1)

    out["ALL"] = all(bool(v) for k, v in out.items() if not k.endswith("_detail"))
    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    r = trap_gate(verbose=True)
    print(f"\ntrap gate (T-3): {'ALL PASS' if r['ALL'] else 'FAIL'}")
