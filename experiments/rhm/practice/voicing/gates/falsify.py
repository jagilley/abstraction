"""[voicing] THE FALSIFICATION HARNESS — the evidence that this node's gate table is not
decorative.

The round's binding rule, adopted after Q2's sixth defect (a gate that read green on a path that
never ran): **a new gate is not reported until it has been shown to FAIL on a deliberate
perturbation of the thing it claims to protect.** This file is where those perturbations live, so
the claim is reproducible instead of a sentence in a halt message.

Q3 added a second rule, after defect #8 (`DESIGN.md` §30). Every gate in this arc had been an
INERTNESS gate — "the treatment, with its knob off, is the control at 0.000e+00" — and that shape
cannot catch a treatment that is not wired at all, because a disconnected treatment is the most
inert thing there is. So a knob whose whole purpose is to change behaviour needs a LIVENESS gate
too, and the perturbation that gate must fail on is DISCONNECTION. V-6 is that gate and its first
perturbation below is the actual defect.

Usage (from experiments/):
    python3 rhm/practice/voicing/gates/falsify.py

Nothing here is imported by a run. Every perturbation monkeypatches a function in `voicing.py`,
checks the gate that guards it goes red, and restores it.
"""
import os, sys, copy, json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))
import numpy as np, torch
import rhm.practice.voicing.voicing as V

res = []
def rec(gate, pert, as_required, detail=""):
    res.append((gate, pert, as_required, detail))
    print(f"  [{'as required' if as_required else 'GATE IS BLIND'}] {gate} << {pert}   {detail}")

# ============================ V-4c: the composed scorer ============================= #
print("\nV-4c (composed scorer)")
orig_compose = V.vo_compose

# P1: drop the weight -> `w = 0` no longer means "the DP alone"
def p_no_w(dp_sc, cr_sc, span, w):
    return orig_compose(dp_sc, cr_sc, span, 1.0)
V.vo_compose = p_no_w
z = V.vo_compose_inert_check(); rec("V-4c", "w is ignored (always 1.0)", not z["ok"],
                                   f"w0_is_dp={z['w0_is_dp']}")

# P2: drop the z-score on the critic -> the composition reads a UNIT, not a preference
def p_no_z(dp_sc, cr_sc, span, w):
    if dp_sc.shape[-1] < 2: return dp_sc
    a = dp_sc / float(span)
    a = (a - a.mean(-1, keepdim=True)) / a.std(-1, keepdim=True).clamp_min(1e-6)
    return a + float(w) * cr_sc
V.vo_compose = p_no_z
z = V.vo_compose_inert_check(); rec("V-4c", "the critic is not z-scored", not z["ok"],
                                   f"affine residual = {z['affine_residual']:.3e}")

# P3: drop the z-score on the DP -> same, one organ over
def p_no_za(dp_sc, cr_sc, span, w):
    if dp_sc.shape[-1] < 2: return dp_sc
    b = (cr_sc - cr_sc.mean(-1, keepdim=True)) / cr_sc.std(-1, keepdim=True).clamp_min(1e-6)
    return dp_sc / float(span) + float(w) * b
V.vo_compose = p_no_za
z = V.vo_compose_inert_check(); rec("V-4c", "the DP is not z-scored", not z["ok"],
                                   f"affine residual = {z['affine_residual']:.3e}")

# P4: drop the |C| = 1 guard -> std 0 on a one-row table
def p_no_guard(dp_sc, cr_sc, span, w):
    a = dp_sc / float(span)
    a = (a - a.mean(-1, keepdim=True)) / a.std(-1, keepdim=True)
    b = (cr_sc - cr_sc.mean(-1, keepdim=True)) / cr_sc.std(-1, keepdim=True)
    return a + float(w) * b
V.vo_compose = p_no_guard
z = V.vo_compose_inert_check(); rec("V-4c", "no |C|=1 guard", not z["ok"],
                                   f"singleton={z['singleton_passthrough']}")
V.vo_compose = orig_compose
z = V.vo_compose_inert_check(); assert z["ok"], z

# ============================ V-4d: the probe path ================================== #
print("\nV-4d (probe path)")
orig_probes = V.vo_run_probes
src = None
import inspect
src_txt = inspect.getsource(orig_probes)

# P1: file into the FILED buffer instead of the probe buffer (the off-stream claim)
def p_into_buf(*a, **k):
    out, n = orig_probes(*a, **k)
    vo = a[0]
    for key, parts in out.items():
        o_ = torch.cat([p[0] for p in parts]); w_ = torch.cat([p[1] for p in parts])
        y_ = torch.cat([p[2] for p in parts]).float()
        o0, w0, y0 = vo.buf[key]
        vo.buf[key] = (torch.cat([o0, o_]), torch.cat([w0, w_]), torch.cat([y0, y_]))
    return out, n
V.vo_run_probes = p_into_buf
z = V.vo_probe_offstream_check(); rec("V-4d", "probe rows pushed into the FILED buffer",
                                      not z["ok"], f"buf_untouched={z['buf_untouched']}")

# P2: bill nothing (the priced claim)
def p_free(*a, **k):
    out, n = orig_probes(*a, **k)
    return out, 0
V.vo_run_probes = p_free
z = V.vo_probe_offstream_check(); rec("V-4d", "the probe is not billed", not z["ok"],
                                      f"billed={z['n_ground']} rows={z['n_rows']}")

# P3: substitute the class that was WRITTEN (the counterfactual claim)
def p_same_class(vo, rows, slots, quot, shared, rules, canon, s, device, *, n_probe,
                 grade_fn, roots_of, **_kw):
    out, n = orig_probes(vo, rows, slots, quot, shared, rules, canon, s, device,
                         n_probe=n_probe, grade_fn=grade_fn, roots_of=roots_of, **_kw)
    for key, parts in out.items():
        mv = slots[key]["move"]
        w = torch.cat([p["w"] for p in rows[key]])
        # [overtone] a part is (obs, cand, y[, draw tag]); keep whatever tail it carries
        out[key] = [(p_[0], w[:p_[1].shape[0]]) + tuple(p_[2:]) for p_ in parts]
    return out, n
V.vo_run_probes = p_same_class
z = V.vo_probe_offstream_check(); rec("V-4d", "the probe re-grades the class that was written",
                                      not z["ok"], f"same-class draws = {z['n_same_class']}")

# P4: off-table candidate (the on-table claim)
def p_offtable(*a, **k):
    out, n = orig_probes(*a, **k)
    for key, parts in out.items():
        out[key] = [(p_[0], (p_[1] + 1) % 999) + tuple(p_[2:]) for p_ in parts]
    return out, n
V.vo_run_probes = p_offtable
z = V.vo_probe_offstream_check(); rec("V-4d", "an off-table candidate", not z["ok"],
                                      f"on_table={z['on_table']}")
V.vo_run_probes = orig_probes
z = V.vo_probe_offstream_check(); assert z["ok"], z

# ====================== V-4A / V-4B / V-5: the in-substrate gates ==================== #
print("\nV-4A / V-4B / V-5 (in-substrate, on synthetic arm files)")
NC = 6
def mk(probe=None, d_fb=1.0):
    g = np.random.default_rng(0)
    log = {k: list(np.round(g.random(NC), 6)) for k in V._VO_SER_V4}
    log["miner"] = [{"2": {"n": i}} for i in range(NC)]
    log["vocab"] = [{"2": 4 + i} for i in range(NC)]
    log["vo"] = [{"critic": {"2n0": {"n": 40, "auc": 0.6}}} for _ in range(NC)]
    base = np.cumsum(np.full(NC, 1000.0))
    pg = list(probe) if probe is not None else [0] * NC
    log["t_cum"] = list(base + np.cumsum(np.asarray(pg, float)) * d_fb)
    log["vo_bill"] = [{"probe_ground": int(x), "probe_priced": float(x * d_fb),
                       "cycle_priced": 1000.0 + x * d_fb,
                       "share": x * d_fb / (1000.0 + x * d_fb)} for x in pg]
    return {"config": {"d_fb": d_fb}, "log": log,
            "events": [{"kind": "commit", "level": 3, "cycle": 4}],
            "vo_probe_buf": ({"2n0": int(sum(pg))} if probe is not None else {})}

A = mk()
B_a = mk()                                   # form A's twin: no probe
B_b = mk(probe=[0, 0, 8, 8, 8, 8])           # form B's twin: probes from c3

def run(a, b, form, label, want_fail):
    try:
        V.vo_gate_v4_v5(copy.deepcopy(a), copy.deepcopy(b), form, arm="synthetic")
        failed = False; det = ""
    except AssertionError as e:
        failed = True; det = str(e)[:110]
    rec(f"V-4{form}", label, failed == want_fail,
        ("RAISED: " + det) if failed else "passed (clean)")
    return failed

# the clean pair must PASS
run(A, B_a, "A", "clean pair (must PASS)", False)
run(A, B_b, "B", "clean pair (must PASS)", False)

# P1: a behaviour series moves
b = copy.deepcopy(B_a); b["log"]["succ"][2] += 1e-9
run(A, b, "A", "one succ value moved by 1e-9", True)
b = copy.deepcopy(B_b); b["log"]["e"][1] += 1e-9
run(A, b, "B", "one e value moved by 1e-9", True)

# P2: a commit moved
b = copy.deepcopy(B_a); b["events"] = [{"kind": "commit", "level": 3, "cycle": 5}]
run(A, b, "A", "the commit cycle moved", True)

# P3: the bill is not the probe count
b = copy.deepcopy(B_b); b["log"]["t_cum"] = [x + 1.0 for x in b["log"]["t_cum"]]
run(A, b, "B", "t_cum off by 1.0 (unbilled cost)", True)
b = copy.deepcopy(B_b); b["log"]["vo_bill"][3]["probe_ground"] = 4
run(A, b, "B", "a probe graded but not counted", True)

# P4: the probe reached the repertoire
b = copy.deepcopy(B_b); b["log"]["miner"][4] = {"2": {"n": 99}}
run(A, b, "B", "the miner state moved", True)
b = copy.deepcopy(B_b); b["log"]["vocab"][3] = {"2": 999}
run(A, b, "B", "the committed vocabulary moved", True)

# P5: vacuity
b = copy.deepcopy(B_a); b["log"]["vo"] = [None] * NC
run(A, b, "A", "the critic never audited (VACUOUS)", True)
b = copy.deepcopy(B_a)
run(A, b, "B", "no probe ever fired (VACUOUS)", True)
b = copy.deepcopy(B_b); b["vo_probe_buf"] = {}
run(A, b, "B", "probes billed but nothing filed (VACUOUS)", True)
# and the filed-critic twin must not be allowed to bill probes
b = copy.deepcopy(B_b)
run(A, b, "A", "form A's twin billed probes", True)

bad = [r for r in res if not r[2]]
print(f"\n{len(res)-len(bad)}/{len(res)} perturbations behaved as required"
      + (f"  BLIND: {bad}" if bad else ""))

# ============================ V-6: the critic's diet knob is LIVE ==================== #
print("\nV-6 (diet knob live) — the gate defect #8 needed")
orig_ct = V.vo_critic_terms

# P1: THE DEFECT ITSELF — the call site drops the keyword, so use_probe takes its default
def p_defect(*a, **k):
    k.pop("use_probe", None)
    return orig_ct(*a, **k)
V.vo_critic_terms = p_defect
z = V.vo_probe_live_check()
rec("V-6", "defect #8: the use_probe keyword is dropped at the call site", not z["live"],
    f"d_critic={z['d_critic']:.3e}  rows {z['n_train_off']}->{z['n_train_on']}")

# P2: the union happens but the probe rows are dropped by the shape guard
def p_shape(*a, **k):
    k["use_probe"] = False
    return orig_ct(*a, **k)
V.vo_critic_terms = p_shape
z = V.vo_probe_live_check()
rec("V-6", "use_probe forced False (a stale shape guard would do the same)", not z["live"],
    f"d_critic={z['d_critic']:.3e}  rows {z['n_train_off']}->{z['n_train_on']}")

V.vo_critic_terms = orig_ct
z = V.vo_probe_live_check()
rec("V-6", "clean (must PASS)", z["live"],
    f"d_critic={z['d_critic']:.3e}  rows {z['n_train_off']}->{z['n_train_on']}")

bad = [r for r in res if not r[2]]
print(f"\nFINAL: {len(res)-len(bad)}/{len(res)} perturbations behaved as required"
      + (f"  BLIND: {bad}" if bad else ""))

# ==================== Y-1 / V-6b: the yoke and the in-substrate diet ================= #
print("\nY-1 (the replay matches) and V-6b (the diet is live in substrate)")

def mk_actions(commits, advances, cancelled=()):
    out = [{"cycle": c, "kind": "commit", "level": 2 + i} for i, c in enumerate(commits)]
    out += [{"cycle": c, "kind": "advance", "level": None} for c in advances]
    out += [{"cycle": c, "kind": "commit", "level": l, "cancelled": True} for c, l in cancelled]
    return out

def mk_arm(commits, advances, cancelled=(), ser=None, gov=True, n_probe=0, rows=0, nc=8):
    g = np.random.default_rng(0)
    log = {k: list(np.round(g.random(nc), 6)) for k in
           ("e","succ","dres","n_solved","n_moves","width","e_practice","gloss","vloss",
            "n_mined","t_cum","g_per_solve")}
    if ser: log.update({k: list(v) for k, v in ser.items()})
    log["vo"] = [{"governed": (["2:0"] if gov else []), "n_probe": n_probe,
                  "critic": {"2n0": {"n": 40, "auc": 0.6}}} for _ in range(nc)]
    return {"config": {"d_fb": 1.0}, "log": log,
            "loop_actions": mk_actions(commits, advances, cancelled),
            "events": [{"kind": "commit", "level": 2 + i, "cycle": c}
                       for i, c in enumerate(commits)],
            "vo_probe_buf": ({"2:0": rows} if rows else {})}

SRC = mk_arm([48, 100, 151, 186], [60, 110, 180, 192, 201])

def runY(yk, label, want_fail):
    try:
        V.vo_gate_yoke(copy.deepcopy(SRC), copy.deepcopy(yk), arm="synthetic", src_name="src")
        failed, det = False, ""
    except AssertionError as e:
        failed, det = True, str(e)[:100]
    rec("Y-1", label, failed == want_fail, ("RAISED: " + det) if failed else "passed (clean)")

runY(mk_arm([48, 100, 151, 186], [60, 110, 180, 192, 201]), "exact replay (must PASS)", False)
runY(mk_arm([48, 100, 151, 186], [60, 110, 180, 192, 201],
            cancelled=[(151, 4)]), "a cancelled commit beside a full match (must PASS)", False)
runY(mk_arm([48, 100, 151], [60, 110, 180, 192, 201]), "one commit missing", True)
runY(mk_arm([48, 100, 151, 187], [60, 110, 180, 192, 201]), "a commit one cycle late", True)
runY(mk_arm([48, 100, 151, 186], [60, 110, 180, 192]), "one advance missing", True)
runY(mk_arm([48, 100, 151, 186], [60, 110, 180, 192, 201, 205]), "an extra advance", True)
try:
    V.vo_gate_yoke(mk_arm([], []), mk_arm([], []), arm="s", src_name="src"); f = False
except AssertionError: f = True
rec("Y-1", "an empty plan (VACUOUS)", f, "RAISED" if f else "passed — GATE IS BLIND")

def runV(a, b, label, want_fail):
    try:
        V.vo_gate_v6b(copy.deepcopy(a), copy.deepcopy(b), "a", "b")
        failed, det = False, ""
    except AssertionError as e:
        failed, det = True, str(e)[:100]
    rec("V-6b", label, failed == want_fail, ("RAISED: " + det) if failed else "passed (clean)")

A6 = mk_arm([48], [60], gov=True)
B6 = mk_arm([48], [60], gov=True, n_probe=64, rows=512,
            ser={"e": [0.5] * 7 + [0.4]})          # one cycle differs: the diet reached the run
runV(A6, B6, "the diet moved one cycle's e (must PASS)", False)
# THE DEFECT: the probe arm is bit-identical to its filed twin on every behaviour series
D6 = mk_arm([48], [60], gov=True, n_probe=64, rows=512)
runV(A6, D6, "defect #8: probes bought, billed, filed — and nothing moved", True)
runV(A6, mk_arm([48], [60], gov=True, n_probe=0, rows=0,
                ser={"e": [0.5] * 7 + [0.4]}), "the probe never fired (VACUOUS)", True)
runV(A6, mk_arm([48], [60], gov=True, n_probe=64, rows=0,
                ser={"e": [0.5] * 7 + [0.4]}), "probes billed, nothing filed (VACUOUS)", True)
runV(mk_arm([48], [60], gov=False), B6, "a twin never governed (VACUOUS)", True)

bad = [r for r in res if not r[2]]
print(f"\nFINAL (Q3 + Q3b): {len(res)-len(bad)}/{len(res)} perturbations behaved as required"
      + (f"  BLIND: {bad}" if bad else ""))
