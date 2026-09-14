"""accelerando Phase 2 — the PROGRAM: a phase-conditioned emitter, and the decider that plays it.

THE ONE LEARNED COMPONENT IN THIS NODE, and it is not a forward model. `ProgHead` maps
`(phase, posture, slot, tempo) -> command`. It predicts no state, it is never rolled forward, and
nothing in this file imports or evaluates an `f(s,u)`. It is `solo/`'s `SpanHead` with two
changes, both forced by the question:

  * **No trunk read.** solo fed the span head its behaviour-cloned trunk's hidden state, because
    the point there was whether corridor content interferes with the executor it shares a body
    with. Here the question is transfer across tempo, so the head takes only what the brief names
    — posture, slot, tempo — and Phase 2 has exactly one learned component.

  * **Phase-conditioned output, not a fixed-width vector.** solo's head emitted `max_span * 2`
    values at once, which is a RECORDING with a learned index: it has one output slot per control
    step and therefore one output shape per tempo. A level-l unit here is `2^(l-1) * H` steps and
    `H` is the era knob, so a fixed-width head could not be asked the same question at two tempi.
    `ProgHead` instead emits `u(phi)` for a scalar phase `phi in [0,1)` along the unit's span and
    is called once per step, so the SAME head answers at every tempo and the number of samples is
    the tempo's business, not the head's. That is the whole difference between a program and a
    recording, made structural rather than asserted: ask for the same trajectory faster and the
    head is still defined; ask a tape and it is not.

WHAT THE HEAD SEES, and why each input is there:
    `phi`      the phase along the unit's span, plus a Fourier expansion of it. An MLP on a raw
               scalar cannot represent a command curve that turns eight times inside one level-4
               unit; `n_freq` sin/cos pairs give it the basis to, at a cost of 2*n_freq inputs.
    `z`        the launch posture at the seam (4-dim, normalised). This is what makes the unit a
               program rather than a tape: the same slot at a different hand-over emits different
               commands.
    `slot`     an embedding over the ladder's stable global slot ids (`LadderLayout`), so a level
               and a seam are named the same way they are everywhere else in the arc.
    `tempo`    `log2(H)`, centred and scaled. Dyadic, so the tempo ladder is EQUALLY SPACED in
               this feature and extrapolating one rung past the practiced set is a unit step —
               which is exactly the extrapolation the parity gate is asked to certify. The head
               also gets `span_notes` so it can tell a level-4 unit from a level-1 one at a fixed
               phase.

`ProgDecider` is `world.LadderDecider` with the content source swapped: the same legality rules,
the same key / audition selection, the same one feedback event at a launch — only the commands
come from the head instead of from a stored member. Keeping the decider identical is what makes
`prog` comparable with `rec` and `scl` row for row.
"""

import numpy as np

from mjc.practice.tempo.accelerando.world import LadderLayout


# --------------------------------------------------------------------------- #
# the deterministic train/held-out split (solo/offbook decision 7, verbatim)
# --------------------------------------------------------------------------- #

def split_code(rows):
    """Deterministic train/held-out split keyed by a BIJECTIVE-ish code of the seam state, not by
    a coin flip: the metering geometries are fixed per run, so near-identical seam states recur
    across cells and a coin flip would put copies of one context on both sides — the parity gate
    would then be reading its own training data (`native/span`'s measured trap, carried through
    `offbook` and `solo`)."""
    q = np.round(np.asarray(rows, np.float64) * 1e4).astype(np.int64)
    code = np.zeros(len(q), np.int64)
    for j in range(q.shape[1]):
        code = (code * 1000003 + (q[:, j] % 1000003)) % (2 ** 61 - 1)
    return (code * 48271) % 100


class ProgBuffer:
    """Per-(slot, tempo) self-imitation store: `(launch state, the commands the AUDITION selected
    for that slot at that state, the tempo, the span)`.

    The target is recomputed by the audition at capture time — the head chases the executor it is
    replacing, not a stale copy of it (solo's `SpanBuffer` contract). Held-out states are split off
    by `split_code` and are never trained on; the parity gate reads only those.
    """

    def __init__(self, hold_frac=0.25):
        self.hold_frac = float(hold_frac)
        self.train, self.hold = {}, {}

    def store(self, sid, H, span_notes, S_in, S_true, C):
        """`S_in` is the head's INPUT posture and `S_true` is where a rollout actually starts.

        They are the same object under `--posture true` (p1's) and differ under `--posture obs`,
        where the input is the Delta-stale read the decider will really be handed at deployment
        while the target and every rollout stay anchored to the true state — a plant rollout
        cannot begin from a posture the body was never in. The split is keyed on `S_true`, which
        identifies the trial regardless of what the agent could see of it.
        """
        if len(S_in) == 0:
            return
        S_in = np.asarray(S_in, np.float32)
        S_true = np.asarray(S_true, np.float32)
        C = np.asarray(C, np.float32)
        is_hold = split_code(S_true) < int(round(self.hold_frac * 100))
        for tgt, sel in ((self.hold, is_hold), (self.train, ~is_hold)):
            if not sel.any():
                continue
            tgt[(int(sid), int(H))] = (S_in[sel], S_true[sel], C[sel], int(span_notes))

    def sizes(self):
        return ({f"{k[0]}@{k[1]}": int(len(v[0])) for k, v in self.train.items()},
                {f"{k[0]}@{k[1]}": int(len(v[0])) for k, v in self.hold.items()})


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #

def phase_feats(H, span_notes, n_freq):
    """The phase basis, assembled once per (tempo, span) and shared by every row that uses it.

    THE FACTORISATION THAT MATTERS. A level-4 unit is eight notes long, so a basis over the SPAN
    phase alone would need frequencies past the note rate before it could represent one turn per
    note — the smoke measured the consequence (`psmoke`, 6 epochs: mse 0.0293 against a target
    mean-square of ~0.04, i.e. the head explained about a quarter of its own targets and
    `prog_L4` came back 20x worse than the tape it was cloning). The piece is factorised as
    notes, so the basis is too: a within-NOTE phase (shared across every note, which is where the
    body's own turn-and-settle shape lives), a SPAN phase (which leg of the figure), and the note
    index. Fourier features on both, because an MLP on a raw scalar cannot turn eight times.

    Columns: [sin/cos(phi_note) * n_freq, phi_note, sin/cos(phi_span) * n_freq, phi_span,
              note_frac, tempo_feat, span_feat]
    """
    L = int(span_notes) * int(H)
    j = np.arange(L, dtype=np.float64)
    ph_n = (j % int(H) + 0.5) / float(H)
    ph_s = (j + 0.5) / float(L)
    nf = (np.floor(j / float(H)) + 0.5) / float(span_notes)
    f = np.arange(1, int(n_freq) + 1, dtype=np.float64)
    cols = [np.sin(2 * np.pi * ph_n[:, None] * f), np.cos(2 * np.pi * ph_n[:, None] * f),
            ph_n[:, None],
            np.sin(2 * np.pi * ph_s[:, None] * f), np.cos(2 * np.pi * ph_s[:, None] * f),
            ph_s[:, None], nf[:, None],
            np.full((L, 1), float(tempo_feat(H))), np.full((L, 1), float(span_feat(span_notes)))]
    return np.concatenate(cols, 1).astype(np.float32)


def feat_dim(n_freq):
    return 4 * int(n_freq) + 5


def _build_prog_head():
    import torch
    import torch.nn as nn

    class ProgHead(nn.Module):
        """`(phase features, posture, slot) -> command`, tanh-squashed to the actuator's range.

        Called once per control step of a unit's span, so its output shape is independent of the
        tempo — the tempo enters as a FEATURE, not as an output width. The alphabet is COMMANDS:
        no arbitrary label is ever a target.
        """

        def __init__(self, n_slots, state_dim=4, act_dim=2, hidden=256, layers=3, emb=32,
                     n_freq=8):
            super().__init__()
            self.n_freq, self.act_dim = int(n_freq), int(act_dim)
            self.slot = nn.Embedding(int(n_slots), int(emb))
            nn.init.normal_(self.slot.weight, 0.0, 0.02)
            din = int(emb) + int(state_dim) + feat_dim(n_freq)
            lyr = [nn.Linear(din, hidden), nn.SiLU()]
            for _ in range(int(layers) - 1):
                lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
            self.trunk = nn.Sequential(*lyr)
            self.out = nn.Linear(hidden, int(act_dim))
            self.act = nn.Tanh()

        def forward(self, feat, z, slot_id):
            h = torch.cat([self.slot(slot_id), z, feat], dim=-1)
            return self.act(self.out(self.trunk(h)))

    return ProgHead


def build_prog(n_slots, seed, device, state_dim=4, act_dim=2, hidden=256, layers=3, emb=32,
               n_freq=8):
    """Mint WITHOUT touching the shared torch stream (`offbook` gate G-R): save the global RNG
    state, construct, restore, then re-initialise every parameter from a dedicated generator, so
    two heads minted in one run are independent of each other and of anything else that draws."""
    import torch
    st = torch.get_rng_state()
    cst = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    head = _build_prog_head()(n_slots, state_dim, act_dim, hidden=hidden, layers=layers,
                              emb=emb, n_freq=n_freq)
    torch.set_rng_state(st)
    if cst is not None:
        torch.cuda.set_rng_state_all(cst)
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in head.parameters():
            if p.dim() >= 2:
                p.copy_(torch.empty(p.shape, dtype=p.dtype).normal_(0.0, 0.02, generator=g))
            else:
                p.zero_()
    return head.to(device)


def tempo_feat(H):
    """`log2(H)` centred on the ladder's midpoint and scaled to ~unit steps. Dyadic tempi are
    equally spaced here, so one rung past the practiced set is a unit step in the head's input —
    the extrapolation the parity gate is asked to certify, expressed as the smallest move the
    feature can make."""
    return (np.log2(np.asarray(H, np.float64)) - 3.0) / 2.0


def span_feat(span_notes):
    return (np.log2(np.asarray(span_notes, np.float64))) / 3.0


def make_prog_fn(head, device, norm, n_freq=8):
    """The numpy interface: `(states, slot_id, H, span_notes) -> (n, span_notes*H, 2)`.

    One batched forward per call; the phase block is built by `phase_feats` and cached per
    (tempo, span), so the head never sees a tempo-dependent shape and the emission costs one
    matrix multiply per launch rather than one per step.
    """
    import torch
    cache = {}

    def fn(states, sid, H, span_notes):
        S = np.asarray(states, np.float32)
        n = len(S)
        L = int(span_notes) * int(H)
        if n == 0:
            return np.zeros((0, L, 2), np.float32)
        key = (int(H), int(span_notes))
        if key not in cache:
            cache[key] = torch.tensor(phase_feats(H, span_notes, n_freq), device=device)
        F = cache[key]
        with torch.no_grad():
            zn = ((S.astype(np.float64) - norm[0]) / norm[1]).astype(np.float32)
            z = torch.tensor(np.repeat(zn, L, axis=0), device=device)
            feat = F.repeat(n, 1)
            sl = torch.full((n * L,), int(sid), dtype=torch.long, device=device)
            y = head(feat, z, sl).float().cpu().numpy()
        return np.clip(y.reshape(n, L, 2), -1.0, 1.0).astype(np.float32)

    return fn


def train_prog(head, buf, norm, device, seed, tempi=None, epochs=300, batch=8192, lr=1e-3,
               n_freq=8, log=None):
    """Self-imitation: regress the audition's own chosen commands onto (phase, posture, slot,
    tempo). Plain MSE on the command alphabet — no state is predicted anywhere.

    Trains on the TRAIN split only and on the PRACTICED tempi only (`tempi`); the buffer holds
    entries at every tempo so the parity gate has held-out launch states everywhere, but a tempo
    outside `tempi` is never a training target — that is what makes the fast rungs extrapolation
    rather than interpolation.

    TWO BASELINES are computed beside the loss so the number is interpretable rather than a
    magnitude: `mse_zero` (predicting no command at all — the target's own mean square) and
    `mse_slotmean` (predicting each (slot, tempo) cell's mean tape, i.e. everything except the
    posture-conditioning the head exists to add). A head that has learned the pool sits near
    `mse_slotmean`; one that has learned the POSTURE sits below it; one that has learned nothing
    sits at `mse_zero`. The pool's own rendition spread is logged next to these by the runner.
    """
    import torch

    keep = None if tempi is None else set(int(x) for x in tempi)
    fe, ze, sle, ye, He, cell = [], [], [], [], [], []
    for (sid, H), (S, S_true, C, sp) in sorted(buf.train.items()):
        if keep is not None and int(H) not in keep:
            continue
        L = int(sp) * int(H)
        n = len(S)
        fe.append(np.tile(phase_feats(H, sp, n_freq), (n, 1)))
        ze.append(np.repeat(((S.astype(np.float64) - norm[0]) / norm[1]).astype(np.float32),
                            L, axis=0))
        sle.append(np.full(n * L, int(sid), np.int64))
        ye.append(C.reshape(n * L, 2).astype(np.float32))
        He.append(np.full(n * L, int(H), np.int64))
        cell.append(np.full(n * L, len(cell), np.int64))
    if not fe:
        return dict(n_rows=0, series=[], final=None, by_tempo={})
    feat = torch.tensor(np.concatenate(fe), device=device)
    z = torch.tensor(np.concatenate(ze), device=device)
    sl = torch.tensor(np.concatenate(sle), device=device)
    y = torch.tensor(np.concatenate(ye), device=device)
    Hs = np.concatenate(He)
    cid = np.concatenate(cell)
    yn = np.concatenate(ye)
    N = len(feat)
    base_zero = float((yn ** 2).mean())
    mu = np.zeros_like(yn)
    for c in np.unique(cid):
        m = cid == c
        mu[m] = yn[m].reshape(-1, yn.shape[-1]).mean(0)
    base_slot = float(((yn - mu) ** 2).mean())
    opt = torch.optim.Adam(head.parameters(), lr=float(lr))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs), eta_min=float(lr) / 20)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    series = []
    head.train()
    for ep in range(int(epochs)):
        perm = torch.randperm(N, generator=g).to(device)
        tot, nb = 0.0, 0
        for i in range(0, N, int(batch)):
            idx = perm[i:i + int(batch)]
            loss = ((head(feat[idx], z[idx], sl[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sch.step()
        series.append(tot / max(1, nb))
        if log is not None and (ep % max(1, int(epochs) // 8) == 0 or ep == int(epochs) - 1):
            log(f"        [head] epoch {ep:4d}  mse {series[-1]:.6f}   "
                f"(zero {base_zero:.6f}, slot-mean {base_slot:.6f}, {N} rows)")
    head.eval()
    by_tempo, by_tempo_base = {}, {}
    with torch.no_grad():
        for H in sorted(set(int(x) for x in Hs)):
            idx = torch.nonzero(torch.tensor(Hs == H, device=device)).squeeze(1)
            pr = head(feat[idx], z[idx], sl[idx])
            by_tempo[int(H)] = float(((pr - y[idx]) ** 2).mean())
            m = Hs == H
            by_tempo_base[int(H)] = dict(zero=float((yn[m] ** 2).mean()),
                                         slot_mean=float(((yn[m] - mu[m]) ** 2).mean()))
    return dict(n_rows=int(N), series=series, final=series[-1], by_tempo=by_tempo,
                baseline=dict(zero=base_zero, slot_mean=base_slot), by_tempo_baseline=by_tempo_base)


# --------------------------------------------------------------------------- #
# the decider: `world.LadderDecider` with the content source swapped
# --------------------------------------------------------------------------- #

class ProgDecider:
    """The seam decision for every `prog` arm.

    Identical to `world.LadderDecider` in legality, selection and cost — the approach plays the
    primitive, a drilled seam pays one feedback event, `key` selects by frozen posture key and
    `audit` auditions on the plant — with ONE difference: the commands come from the head, called
    with the observed hand-over, rather than from a stored member.

    `open_slots` is the parity gate's verdict. When it is given, a slot below parity falls back to
    `fallback_lib`'s own best member (the tape), so the gate can never introduce drift — solo's
    contract. When it is `None` the head plays everywhere and the arm measures TRANSFER rather
    than adoption; both arms are reported.
    """

    def __init__(self, W, lib, prog_fn, H, mode="key", levels=(1,), obs_delay=0,
                 open_slots=None, fallback_lib=None, capture=False):
        self.W, self.lib, self.prog_fn = W, lib, prog_fn
        self.H = int(H)
        self.mode = str(mode)
        self.levels = tuple(int(x) for x in levels)
        self.D = int(obs_delay)
        self.open_slots = (None if open_slots is None else set(int(x) for x in open_slots))
        self.fallback_lib = fallback_lib
        self.capture = bool(capture)
        self.picks = []
        self.n_fallback = 0
        self.n_head = 0

    def _legal(self, d):
        out = []
        for ell in sorted(set(self.levels), reverse=True):
            for s in self.lib.slots(ell, d, with_poison=False):
                out.append((int(ell), s))
        return out

    def _content(self, ell, sl, states):
        """The head's emission for this slot, or the tape when the slot is below parity."""
        sp = int(LadderLayout.span(ell))
        sid = int(sl["slot"])
        if self.open_slots is not None and sid not in self.open_slots:
            self.n_fallback += len(states)
            src = self.fallback_lib if self.fallback_lib is not None else self.lib
            fs = src.slot_by_id(sid)
            mm = (fs or sl)["members"]
            return np.tile(np.asarray(mm[0]["cmds"], np.float32)[None], (len(states), 1, 1)), sp
        self.n_head += len(states)
        return self.prog_fn(states, sid, self.H, sp), sp

    def __call__(self, k, states, need, rng, led):
        W, H = self.W, self.W.H
        n = len(states)
        d = int(k) - int(W.k0)
        if d < 0:
            return dict(kind="reflex", span=np.ones(n, int), src="approach")
        legal = self._legal(d)
        if not legal:
            return dict(kind="reflex", span=np.ones(n, int), src="no_content")
        maxspan = max(int(LadderLayout.span(e)) for e, _ in legal)
        out_c = np.zeros((n, maxspan * H, 2), np.float32)
        out_ns = np.ones(n, int)
        chosen, lvl = [], []
        n_aud = 0.0
        if self.mode == "key":
            keys = np.array([s["key"] for _, s in legal], np.float32)
            sd = keys.std(0) + 1e-6
            pick = np.argmin(np.linalg.norm((states[:, None, :] - keys[None]) / sd[None, None],
                                            axis=2), axis=1)
            for ci, (ell, sl) in enumerate(legal):
                sel = np.where(pick == ci)[0]
                if not len(sel):
                    continue
                c, sp = self._content(ell, sl, states[sel])
                out_c[np.ix_(sel, np.arange(sp * H))] = c
                out_ns[sel] = sp
            for i in range(n):
                ell, sl = legal[int(pick[i])]
                chosen.append(int(sl["slot"])); lvl.append(int(ell))
        else:
            scores = np.full((n, len(legal)), np.inf, np.float64)
            cache = {}
            for ci, (ell, sl) in enumerate(legal):
                c, sp = self._content(ell, sl, states)
                errs, _ = W.rollout(states, c, k, sp, led)
                scores[:, ci] = errs.mean(1)
                cache[ci] = (c, sp)
                n_aud += 1.0
            best = np.argmin(scores, axis=1)
            for i in range(n):
                ci = int(best[i])
                ell, sl = legal[ci]
                c, sp = cache[ci]
                out_c[i, :sp * H] = c[i]
                out_ns[i] = sp
                chosen.append(int(sl["slot"])); lvl.append(int(ell))
        lv = np.asarray(lvl, int)
        self.picks.append(dict(seam=int(k), drilled=int(d), n_legal=len(legal),
                               n_aud=float(n_aud), need=[int(b) for b in np.asarray(need, int)],
                               slot=[int(x) for x in chosen], member=[-1] * n,
                               level=[int(x) for x in lvl], mean_level=float(lv.mean()),
                               frac_by_level={str(e): float((lv == e).mean())
                                              for e in sorted(set(lvl))},
                               ns=[int(x) for x in out_ns]))
        return dict(kind="open", cmds=out_c, span=out_ns, src=f"prog_{self.mode}",
                    n_cand=float(len(legal)), frac_prim=0.0, mean_level=float(lv.mean()))


# --------------------------------------------------------------------------- #
# the parity gate
# --------------------------------------------------------------------------- #

def ref_errors(W, buf, ref_lib, H, min_hold, np_=np, led=None):
    """The PARITY REFERENCE: for every (slot, tempo) cell with held-out launch states, the error
    the reference library's own AUDITION achieves at each of those states — its best member,
    executed. solo's gate is "at least as good as the audition's own chosen member", so the
    reference has to be the per-state winner, not a fixed member.

    Computed once per (slot, tempo, library) and reused across heads. Not charged: it is the
    instrument the head is measured against, not a cost the agent pays every cycle.
    """
    ref = {}
    for (sid, Hh), (S_in, S, C, sp) in sorted(buf.hold.items()):
        if int(Hh) != int(H) or len(S) < int(min_hold):
            continue
        sl = ref_lib.slot_by_id(int(sid))
        if sl is None:
            continue
        ell, d, _ = ref_lib.layout.cell_of(int(sid))
        kw = int(W.k0) + int(d)
        mm = sl["members"]
        n, m = len(S), len(mm)
        s0 = np_.repeat(S, m, axis=0)
        cm = np_.tile(np_.stack([b["cmds"] for b in mm]), (n, 1, 1))
        errs, _ = W.rollout(s0, cm, kw, int(sp), led, charge=False)
        ref[int(sid)] = errs.mean(1).reshape(n, m).min(1)
    return ref


def parity(W, buf, prog_fn, ref_lib, ref_err, H, tau, min_hold, band, np_=np, led=None,
           ref_name="rec"):
    """PARITY, measured as execution reproduction ON THE PLANT, per (slot, tempo), on HELD-OUT
    launch states — solo's gate, with the tempo added to the index.

    A slot is OPEN at a tempo iff the head's single emission, executed, is at least as good as the
    reference tape's, executed, on at least `tau` of that slot's held-out states. The reference is
    an argument because at a tempo the learner has never practiced there IS no same-tempo
    recording — `ref_name = "rec"` is the oracle reading (what practice there would have bought)
    and `ref_name = "scl0"` is the deployable one (the best content a learner without practice at
    this tempo actually has). Both are run and both are reported.

    The reference side (`ref_errors`) is an instrument and is not charged; the HEAD side is a real
    self-check the agent performs and is charged. Continuous fractions are logged so another tau
    can be read off the record.
    """
    opens, rows = set(), []
    for (sid, Hh), (S_in, S, C, sp) in sorted(buf.hold.items()):
        if int(Hh) != int(H) or len(S) < int(min_hold) or int(sid) not in ref_err:
            continue
        ell, d, _ = ref_lib.layout.cell_of(int(sid))
        kw = int(W.k0) + int(d)
        # the head is asked with the posture it will be handed at deployment; the rollout starts
        # from the true state, because that is where the body is.
        head = prog_fn(S_in, int(sid), int(H), int(sp))
        e_head, _ = W.rollout(S, head, kw, int(sp), led)
        et, eh = ref_err[int(sid)], e_head.mean(1)
        frac = float(np_.mean(eh <= et))
        rows.append(dict(slot=int(sid), level=int(ell), drilled=int(d), H=int(H),
                         ref=ref_name, n_hold=int(len(S)), frac=frac,
                         mean_e_tape=float(et.mean()), mean_e_head=float(eh.mean()),
                         mean_gap=float((eh - et).mean()),
                         head_in_band=float(np_.mean(eh <= band)),
                         tape_in_band=float(np_.mean(et <= band)),
                         open=bool(frac >= float(tau))))
        if frac >= float(tau):
            opens.add(int(sid))
    return opens, rows
