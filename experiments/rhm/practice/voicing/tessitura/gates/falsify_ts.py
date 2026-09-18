"""[tessitura] THE FALSIFICATION HARNESS FOR THE VALUE-READER ROUND.

`voicing/gates/falsify.py` is the parent harness and its rule is inherited verbatim: **a gate
is not reported until it has been shown to FAIL on a deliberate perturbation of the thing it
claims to protect**, and for any knob whose whole purpose is to change behaviour, the
perturbation it must fail on is DISCONNECTION (`voicing/DESIGN.md` §30).

This round adds five offline gates (`voicing.py::ts_gates_cpu`). Every one of them is perturbed
here. Two further gates are not offline and are not here: TS-1, the full-scale inertness twin
(the paid arm against banked `ov_s0b:ovt_comp_pr_sh` on every behaviour series) and G-F (the
donor replay at 0.000e+00); both need a GPU and a volume, and their records live in
`tessitura/results/`.

Usage (from experiments/):
    python3 rhm/practice/voicing/tessitura/gates/falsify_ts.py

Nothing here is imported by a run. Every perturbation monkeypatches a function in `voicing.py`,
checks the gate that guards it goes red, and restores it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..")))
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import rhm.practice.voicing.voicing as V                                 # noqa: E402
import rhm.practice.fourwall.wall as W                                   # noqa: E402

res = []


def rec(gate, pert, as_required, detail=""):
    res.append((gate, pert, as_required, detail))
    print(f"  [{'as required' if as_required else 'GATE IS BLIND'}] {gate} << {pert}   "
          f"{detail}")


def one(name):
    """Run the offline table quietly and return the record of the gate whose key starts with
    `name`. A perturbation that makes the gate's own fixture RAISE counts as the gate going
    red and is reported as such: a run that crashes is a run that did not pass."""
    try:
        r = V.ts_gates_cpu(verbose=False)
    except Exception as e:                     # noqa: BLE001 — a raise IS a red gate
        return {"pass": False, "detail": f"RAISED {type(e).__name__}: {str(e)[:70]}"}
    for k, q in r.items():
        if k != "ALL" and k.startswith(name):
            return q
    raise KeyError(name)


# ===================== TS-2: the judge's level, added and correct ==================== #
print("\nTS-2 (the level is added beside the audit's fields, is the mean sigmoid, draws nothing)")
_orig_audit = V.vo_critic_audit
_orig_scores = V.vo_critic_scores


def _audit_level_is_logit(*a, **k):
    """The level reported as the mean LOGIT instead of the mean probability — the single most
    likely way to get this quantity wrong, since the critic emits logits."""
    out = _orig_audit(*a, **k)
    for _k, q in out.items():
        if isinstance(q, dict) and "ts" in q:
            q["ts"]["mean_p"] = float(np.log(max(q["ts"]["mean_p"], 1e-9)))
    return out


V.vo_critic_audit = _audit_level_is_logit
r = one("TS-2")
rec("TS-2", "the level reported as a mean logit, not a mean probability", not r["pass"],
    r["detail"][:110])
V.vo_critic_audit = _orig_audit


def _audit_overwrites(*a, **k):
    """The level DISPLACES a pre-existing field — the inertness half."""
    out = _orig_audit(*a, **k)
    for _k, q in out.items():
        if isinstance(q, dict) and "ts" in q:
            q["auc"] = q["ts"]["mean_p"]
    return out


V.vo_critic_audit = _audit_overwrites
r = one("TS-2")
rec("TS-2", "the level overwrites the audit's own `auc`", not r["pass"], r["detail"][:110])
V.vo_critic_audit = _orig_audit


def _audit_draws(*a, **k):
    """The instrument consumes a draw — the RNG-neutrality half. On this file an instrument
    that moves the shared stream moves every arm downstream of it."""
    out = _orig_audit(*a, **k)
    torch.randn(1)
    return out


V.vo_critic_audit = _audit_draws
r = one("TS-2")
rec("TS-2", "the audit consumes a draw", not r["pass"], r["detail"][:110])
V.vo_critic_audit = _orig_audit


# ===================== TS-3: the panel is frozen, the reader is live ================= #
print("\nTS-3 (the panel freezes once per (slot, era); the reading follows the live critic)")
_orig_panel = V.ts_panel_audit


def _panel_refreezes(core, critic, vo, slots, device, *, era, cap, chunk):
    """The panel re-frozen on every call — which is what it would silently become if the
    freeze condition were written on the buffer rather than on the panel's own key. The whole
    protocol dies: the base rate follows the world and no drift is attributable."""
    if hasattr(vo, "ts_panel"):
        vo.ts_panel = {k: q for k, q in vo.ts_panel.items()
                       if int(k.split("|e")[1]) != int(era)}
    return _orig_panel(core, critic, vo, slots, device, era=era, cap=cap, chunk=chunk)


V.ts_panel_audit = _panel_refreezes
r = one("TS-3")
rec("TS-3", "the panel re-freezes from the current buffer on every call", not r["pass"],
    r["detail"][:110])
V.ts_panel_audit = _orig_panel


def _panel_dead(core, critic, vo, slots, device, *, era, cap, chunk):
    """DISCONNECTION: the panel scored by a critic frozen at its own first sight, so the
    reading can never move. This is the perturbation the arc demands of any instrument whose
    purpose is to track something."""
    if not hasattr(vo, "_ts_frozen_critic"):
        import copy as _c
        vo._ts_frozen_critic = _c.deepcopy(critic)
    return _orig_panel(core, vo._ts_frozen_critic, vo, slots, device, era=era, cap=cap,
                       chunk=chunk)


V.ts_panel_audit = _panel_dead
r = one("TS-3")
rec("TS-3", "the panel scored by a critic frozen at first sight (disconnection)",
    not r["pass"], r["detail"][:110])
V.ts_panel_audit = _orig_panel


# ===================== TS-4: the mask does not matter =============================== #
print("\nTS-4 (the structural label is invariant to the span mask, and only to that span)")
_orig_cf = W.consistent_features


def _cf_reads_the_mask(rules, x_np, roots_np, node, level, s, canon_np, v):
    """A label that looks at the masked positions instead of overwriting them — the failure
    mode the gate exists for, and the one that would make `structure.py`'s whole premise
    false."""
    mask, n = _orig_cf(rules, x_np, roots_np, node, level, s, canon_np, v)
    bad = (np.asarray(x_np) < 0).any(1)
    mask[bad] = False
    return mask, n


W.consistent_features = _cf_reads_the_mask
r = one("TS-4")
rec("TS-4", "a label that reads the mask instead of overwriting it", not r["pass"],
    r["detail"][:110])
W.consistent_features = _orig_cf


def _cf_ignores_root(rules, x_np, roots_np, node, level, s, canon_np, v):
    """The root permuted: the label computed against a world that was not the one drawn. The
    gate's masked/unmasked identity survives this (both halves move together), so this
    perturbation is recorded as the one TS-4 is BLIND to, which is exactly why gate T-4 in
    `tessitura/structure.py` exists and fails."""
    return _orig_cf(rules, x_np, (np.asarray(roots_np) + 1) % int(v), node, level, s,
                    canon_np, v)


W.consistent_features = _cf_ignores_root
r = one("TS-4")
rec("TS-4", "the root permuted (EXPECTED to be invisible here — see T-4 in structure.py)",
    not r["pass"],
    (r["detail"][:90] + "  <- blind by design; the root is T-4's subject, not TS-4's")
    if r["pass"] else r["detail"][:110])
W.consistent_features = _orig_cf


# ===================== TS-5: the structural audit is an instrument ================== #
print("\nTS-5 (the audit produces the repair-set membership, counts its reads, draws nothing)")
_orig_struct = V.ts_struct_audit


def _struct_uncounted(*a, **k):
    """The oracle reads not counted — the accounting half. `rep`'s own rule is that a grammar
    read is a read even when nothing is billed for it."""
    before = int(V._EXP_REC["reads"])
    out = _orig_struct(*a, **k)
    V._EXP_REC["reads"] = before
    return out


V.ts_struct_audit = _struct_uncounted
r = one("TS-5")
rec("TS-5", "the audit's grammar reads are not counted", not r["pass"], r["detail"][:110])
V.ts_struct_audit = _orig_struct


def _struct_wrong_label(*a, **k):
    """The structural label replaced by the verdict — the collapse the whole question is about.
    If the gate cannot see the two labels being the same object, nothing downstream of it can."""
    out = _orig_struct(*a, **k)
    for _k, q in out.items():
        q["base_struct"] = q["base_y"]
    return out


V.ts_struct_audit = _struct_wrong_label
r = one("TS-5")
rec("TS-5", "the structural label replaced by the verdict", not r["pass"], r["detail"][:110])
V.ts_struct_audit = _orig_struct


def _struct_draws(*a, **k):
    out = _orig_struct(*a, **k)
    torch.randn(1)
    return out


V.ts_struct_audit = _struct_draws
r = one("TS-5")
rec("TS-5", "the audit consumes a draw", not r["pass"], r["detail"][:110])
V.ts_struct_audit = _orig_struct


# ===================== TS-6: the probe's fifth element is gated ===================== #
print("\nTS-6 (the root rides the probe tuples only when the knob is on; the buffer is inert)")
_orig_probes = V.vo_run_probes


def _probes_ungated(vo, rows, *a, **k):
    """The root appended UNCONDITIONALLY — the change as it would look if the knob were
    forgotten. Harmless in fact, which is exactly why it needs a gate rather than an argument."""
    out, ng = _orig_probes(vo, rows, *a, **k)
    for key, parts in out.items():
        fixed = []
        for p in parts:
            fixed.append(p if len(p) > 4 else p + (torch.zeros(int(p[2].shape[0]),
                                                               dtype=torch.long),))
        out[key] = fixed
    return out, ng


V.vo_run_probes = _probes_ungated
r = one("TS-6")
rec("TS-6", "the root appended unconditionally", not r["pass"], r["detail"][:110])
V.vo_run_probes = _orig_probes


_orig_push = V.VoRecorder.push_probe


def _push_reads_five(self, rows):
    """`push_probe` made to read the fifth element into the verdict column — the one way the
    added element could reach the critic's diet. The probe buffer must then differ between the
    two knob settings, and the gate must see it."""
    fixed = {}
    for key, parts in rows.items():
        fixed[key] = [(p[0], p[1], (p[4].float() if len(p) > 4 else p[2]), p[3])
                      for p in parts]
    return _orig_push(self, fixed)


V.VoRecorder.push_probe = _push_reads_five
r = one("TS-6")
rec("TS-6", "`push_probe` reads the fifth element into the verdict", not r["pass"],
    r["detail"][:110])
V.VoRecorder.push_probe = _orig_push


# ============================== the tally ========================================== #
blind = [q for q in res if not q[2]]
expected_blind = [q for q in res if not q[2] and "EXPECTED to be invisible" in q[1]]
print(f"\n[tessitura] {len(res) - len(blind)}/{len(res)} perturbations turn their gate red"
      + (f"; {len(expected_blind)} of the remainder are blind BY DESIGN and named as such"
         if expected_blind else ""))
for g, p_, okq, d in res:
    if not okq and "EXPECTED to be invisible" not in p_:
        print(f"  !! {g} did not go red on: {p_}   {d}")
assert all(q[2] or "EXPECTED to be invisible" in q[1] for q in res), \
    "a gate stayed green under a perturbation of what it protects"
print("[tessitura] falsification harness: OK")
