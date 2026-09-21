"""[aliquot] THE FALSIFICATION HARNESS — `sotto_voce/gates/falsify.py` forked and extended.

The evidence that this node's gate table is not decorative.

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

[sotto] adds the mirror's gates. M-1 / M-3 / M-4 ride on `vo_mirror_check`, M-1b on
`vo_mirror_loss_check`, M-2 / M-2b on `vo_mirror_source_check` and M-5 on `vo_om_live_check`,
each shown to fail on a deliberate perturbation of exactly what it claims to protect: wiring
the world's verdict into `pbuf`, feeding a model-graded row to the outcome models, billing a
model-graded probe, flipping the agreement threshold, and a mirror that is never trained.

[aliquot] adds the projection's gates. P-1 / P-2 / P-3 / P-4 ride on `vo_proj_check` and
P-2r / P-3r / P-4r on `vo_gate_proj_run`, each shown to fail on a deliberate perturbation of
exactly what it claims to protect: a readout that is not linear in the state, a fit that steps
the plant, a fit that consumes the shared RNG stream, features CACHED instead of recomputed
against the moving trunk, a "random" twin that is secretly the live trunk, a twin that is
trained, a readout that is never refit, and the two run-level forms of the live/frozen contrast.

Usage (from experiments/, or `modal run .../soundboard.py::falsify_remote` where torch is):
    python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/gates/falsify.py

Nothing here is imported by a run. Every perturbation monkeypatches a function in
`aliquot.py`, checks the gate that guards it goes red, and restores it.
"""
import os, sys, copy, json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "..", "..", "..")))
import numpy as np, torch
import rhm.practice.voicing.sotto_voce.aliquot.soundboard.soundboard as V

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
    out, n, wr = orig_probes(*a, **k)
    vo = a[0]
    for key, parts in out.items():
        o_ = torch.cat([p[0] for p in parts]); w_ = torch.cat([p[1] for p in parts])
        y_ = torch.cat([p[2] for p in parts]).float()
        o0, w0, y0 = vo.buf[key]
        vo.buf[key] = (torch.cat([o0, o_]), torch.cat([w0, w_]), torch.cat([y0, y_]))
    return out, n, wr
V.vo_run_probes = p_into_buf
z = V.vo_probe_offstream_check(); rec("V-4d", "probe rows pushed into the FILED buffer",
                                      not z["ok"], f"buf_untouched={z['buf_untouched']}")

# P2: bill nothing (the priced claim)
def p_free(*a, **k):
    out, n, wr = orig_probes(*a, **k)
    return out, 0, wr
V.vo_run_probes = p_free
z = V.vo_probe_offstream_check(); rec("V-4d", "the probe is not billed", not z["ok"],
                                      f"billed={z['n_ground']} rows={z['n_rows']}")

# P3: substitute the class that was WRITTEN (the counterfactual claim)
def p_same_class(vo, rows, slots, quot, shared, rules, canon, s, device, *, n_probe,
                 grade_fn, roots_of, **kk):
    out, n, wr = orig_probes(vo, rows, slots, quot, shared, rules, canon, s, device,
                             n_probe=n_probe, grade_fn=grade_fn, roots_of=roots_of, **kk)
    for key, parts in out.items():
        w = torch.cat([p["w"] for p in rows[key]])
        out[key] = [(o_, w[:c_.shape[0]], y_, yw_) for (o_, c_, y_, yw_) in parts]
    return out, n, wr
V.vo_run_probes = p_same_class
z = V.vo_probe_offstream_check(); rec("V-4d", "the probe re-grades the class that was written",
                                      not z["ok"], f"same-class draws = {z['n_same_class']}")

# P4: off-table candidate (the on-table claim)
def p_offtable(*a, **k):
    out, n, wr = orig_probes(*a, **k)
    for key, parts in out.items():
        out[key] = [(o_, (c_ + 1) % 999, y_, yw_) for (o_, c_, y_, yw_) in parts]
    return out, n, wr
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


# ======================= [sotto] M-1 / M-3 / M-4: the mirror's dispatch ============== #
print("\n[sotto] M-1 / M-3 / M-4 (who answers, who is billed, what gets filed)")
orig_probes2 = V.vo_run_probes

for _m in ("model", "committee", "hybrid"):
    z = V.vo_mirror_check(_m)
    rec("M-1/M-3/M-4", f"clean, mode={_m} (must PASS)", z["ok"],
        f"filed {z['n_filed']}/{z['want_filed']} billed {z['billed']}/{z['want_bill']}")

# P1 (M-1): wire the WORLD's verdict into `pbuf` — the exact perturbation the spec names
def p_world_filed(*a, **k):
    out, n, wr = orig_probes2(*a, **k)
    for key, parts in out.items():
        out[key] = [(o_, c_, yw_.clone(), yw_) for (o_, c_, y_, yw_) in parts]
    return out, n, wr
V.vo_run_probes = p_world_filed
z = V.vo_mirror_check("model")
rec("M-1", "the world's verdict is filed into pbuf on a model-graded arm", not z["ok"],
    f"filed verdicts that are the world's = {z['filed_verdicts_that_are_the_world_s']} "
    f"(want {z['want_filed_verdicts_that_are_the_world_s']})")

# P2 (M-1, vacuity): the world is never consulted at all, so the instrument is empty
def p_no_world(*a, **k):
    out, n, wr = orig_probes2(*a, **k)
    vo = a[0]
    vo.mirror.clear()
    return out, n, wr
V.vo_run_probes = p_no_world
z = V.vo_mirror_check("model")
rec("M-1", "the world's verdict is never recorded (VACUOUS instrument)", not z["ok"],
    f"instrument rows = {z['mirror_rows']} of {z['n_probe']}")

# P3 (M-3): bill the model-graded probes
def p_bill_model(*a, **k):
    out, n, wr = orig_probes2(*a, **k)
    return out, n + sum(int(p[1].shape[0]) for parts in out.values() for p in parts), wr
V.vo_run_probes = p_bill_model
z = V.vo_mirror_check("model")
rec("M-3", "a model-graded probe is billed", not z["ok"],
    f"billed {z['billed']} (want {z['want_bill']})")

# P4 (M-3): the hybrid bills every probe, not only the disagreeing ones
V.vo_run_probes = p_bill_model
z = V.vo_mirror_check("hybrid")
rec("M-3", "the hybrid bills the agreeing probes too", not z["ok"],
    f"billed {z['billed']} (want {z['want_bill']})")
V.vo_run_probes = orig_probes2

# P5 (M-4): FLIP THE THRESHOLD — the spec's own perturbation. With the threshold above every
# spread the committee arm files everything, which is the model-graded arm wearing its name.
z = V.vo_mirror_check("committee", thr=1e9)
rec("M-4", "the agreement threshold flipped wide open (committee files everything)",
    not z["ok"], f"filed {z['n_filed']} (want {z['want_filed']})")
z = V.vo_mirror_check("committee", thr=-1.0)
rec("M-4", "the agreement threshold flipped shut (committee files nothing)", not z["ok"],
    f"filed {z['n_filed']} (want {z['want_filed']})")
z = V.vo_mirror_check("hybrid", thr=-1.0)
rec("M-4", "the threshold shut on the hybrid (every probe goes to the world)", not z["ok"],
    f"billed {z['billed']} (want {z['want_bill']})")

# ======================= [sotto] M-1b: the world's verdict enters no loss ============ #
print("\n[sotto] M-1b (the world's verdict enters NO LOSS)")
z = V.vo_mirror_loss_check()
rec("M-1b", "clean (must PASS)", z["ok"],
    f"d = {z['d_critic_between_world_buffers']:.3e}  step moved {z['d_step']:.3e}")

orig_terms = V.vo_critic_terms
_VOCT = V.vo_critic_terms

def p_read_wbuf(core, critic, ex, slots, s, batch, rng, device, **k):
    """THE PERTURBATION: the critic's loss reads the WORLD buffer instead of the filed one."""
    vo = getattr(ex, "vo", None)
    keep = {}
    for key in list(vo.pbuf):
        if vo.wbuf.get(key) is not None:
            keep[key] = vo.pbuf[key]
            o_, c_, _y = vo.pbuf[key]
            vo.pbuf[key] = (o_, c_, vo.wbuf[key].clone())
    out = _VOCT(core, critic, ex, slots, s, batch, rng, device, **k)
    for key, val in keep.items():
        vo.pbuf[key] = val
    return out
V.vo_critic_terms = p_read_wbuf
z = V.vo_mirror_loss_check()
rec("M-1b", "the critic's loss reads the WORLD buffer", not z["ok"],
    f"d = {z['d_critic_between_world_buffers']:.3e}")
V.vo_critic_terms = orig_terms

# ======================= [sotto] M-2: the outcome models' sources ==================== #
print("\n[sotto] M-2 / M-2b (the outcome models train only on their stated sources)")
z = V.vo_mirror_source_check()
rec("M-2", "clean (must PASS)", z["ok"],
    f"hybrid admitted {z['hybrid']['n_push_world']} world rows, model {z['model']['n_push_world']}")

orig_pw = V.VoOutcomeBank.push_world

# P1: open the door on every arm — a model-graded probe row can now reach the models
def p_open_door(self, x, r, y, blk=None, spn=None):      # [aliquot] the slot columns
    self._append(x, r, y, 1, blk=blk, spn=spn)
    self.n_push_world += int(x.shape[0])
    return int(x.shape[0])
V.VoOutcomeBank.push_world = p_open_door
z = V.vo_mirror_source_check()
rec("M-2", "the door is opened on every arm (a model-graded probe row is fed in)",
    not z["ok"],
    f"model arm admitted {z['model']['n_push_world']} probe rows (must be 0)")
V.VoOutcomeBank.push_world = orig_pw

# P2: the BACK door — a row appended around `push_world`, so the counter says zero
orig_app = V.VoOutcomeBank._append

def p_backdoor(self, x, r, y, src, blk=None, spn=None, yq=None, tid=None):
    # [aliquot] the slot columns; [soundboard] the yield/trajectory columns
    orig_app(self, x, r, y, src, blk=blk, spn=spn, yq=yq, tid=tid)
    if self.mode != "hybrid" and int(src) == 0 and int(x.shape[0]) > 4:
        orig_app(self, x[:4], r[:4], 1.0 - y[:4], 1)      # smuggled in, counter untouched
V.VoOutcomeBank._append = p_backdoor
z = V.vo_mirror_source_check()
rec("M-2", "a world-marked row appended AROUND the door (the counter says zero)",
    not z["ok"],
    f"model arm carries {z['model']['n_src_world']} world rows but admitted "
    f"{z['model']['n_push_world']}")
V.VoOutcomeBank._append = orig_app

# P3 (M-2b): a training row whose verdict is not the world's on its own configuration
orig_pe = V.VoOutcomeBank.push_experience

def p_wrong_y(self, rows, roots_np, slots=None, label_fn=None):
    # [aliquot] the slot map; [soundboard] the yield label hook
    n = orig_pe(self, rows, roots_np, slots, label_fn=label_fn)
    if n:
        self.y[-1] = 1.0 - self.y[-1]
    return n
V.VoOutcomeBank.push_experience = p_wrong_y
z = V.vo_mirror_source_check()
rec("M-2b", "one training row's verdict is not the world's on its configuration",
    not z["ok"],
    f"re-grade mismatches = {sum(z[k]['regrade_bad'] for k in ('model','committee','hybrid'))}")
V.VoOutcomeBank.push_experience = orig_pe

# ======================= [sotto] M-5: the mirror is LIVE ============================= #
print("\n[sotto] M-5 (the mirror is live)")
z = V.vo_om_live_check()
rec("M-5", "clean (must PASS)", z["ok"],
    f"d_params={z['d_params']:.3e} member gap={z['member_gap']:.4f} auc={z['hold_auc']:.3f}")

orig_train = V.VoOutcomeBank.train

# P1: THE DISCONNECTION PERTURBATION, defect #8's shape one organ over — the mirror is built,
# fed, consulted, and never trained.
def p_no_train(self):
    self.stat["n_rows"] = int(self.x.shape[0])
    self.stat["n_hold"] = int((self.h < self.hold).sum())
    self.stat["trained"] += 1
    return 0.0
V.VoOutcomeBank.train = p_no_train
z = V.vo_om_live_check()
rec("M-5", "the mirror is never trained (DISCONNECTION)", not z["ok"],
    f"d_params = {z['d_params']:.3e}")
V.VoOutcomeBank.train = orig_train

# P2: one committee, one seed — the members are copies and the spread is identically zero
orig_bo = V.build_outcome
V.build_outcome = lambda v_, L_, s_, dim_, seed_, dev_: orig_bo(v_, L_, s_, dim_, 1, dev_)
orig_ap2 = V.VoOutcomeBank._append

def p_same_boot(self, x, r, y, src, blk=None, spn=None, yq=None, tid=None):
    # [aliquot] the slot columns; [soundboard] the yield/trajectory columns
    orig_ap2(self, x, r, y, src, blk=blk, spn=spn, yq=yq, tid=tid)
    self.u[:] = 0.0                       # every member gets every row: same init, same data
V.VoOutcomeBank._append = p_same_boot
_saved_mrng = V.VoOutcomeBank.__init__

def p_same_draw(self, cfg, v_, length, s_, device, seed):
    _saved_mrng(self, cfg, v_, length, s_, device, seed)
    self.mrng = [np.random.default_rng(1) for _ in range(self.K)]
V.VoOutcomeBank.__init__ = p_same_draw
z = V.vo_om_live_check()
rec("M-5", "the committee is K copies of one member (spread identically zero)", not z["ok"],
    f"member gap = {z['member_gap']:.6f}")
V.VoOutcomeBank.__init__ = _saved_mrng
V.VoOutcomeBank._append = orig_ap2
V.build_outcome = orig_bo
z = V.vo_om_live_check()
rec("M-5", "restored (must PASS)", z["ok"], f"member gap = {z['member_gap']:.4f}")

# ============ [aliquot] P-1 / P-2 / P-3 / P-4: the projection and its twin ============ #
print("\n[aliquot] P-1/P-2/P-3/P-4 (the projection is a linear readout of a LIVE trunk)")
z = V.vo_proj_check()
rec("P-1..P-5", "clean (must PASS)", z["ok"],
    f"affine err={z['P-1:affine_err']:.2e} live|dp|={z['P-3:live_features_moved']:.3e} "
    f"twin|dp|={z['P-3:twin_features_still']:.3e} refits={z['P-4:n_refit']}")

# P1: the readout is NOT linear in the state — a square term is smuggled into the link
orig_score = V.VoProjBank._score
def p_nonlinear(self, Z, w):
    import torch
    with torch.no_grad():
        e = Z.float().to(self.device) @ w.float()
        return torch.sigmoid(e + 0.35 * e * e).cpu()
V.VoProjBank._score = p_nonlinear
z = V.vo_proj_check()
rec("P-1", "the link is not affine in the pooled state (a square term smuggled in)",
    not z["P-1:affine"], f"affine residual = {z['P-1:affine_err']:.3e}")
V.VoProjBank._score = orig_score

# P2: the FIT STEPS THE PLANT — the readout stops being a read and becomes a treatment
orig_tr = V.VoProjBank.train
def p_steps_plant(self):
    import torch
    out = orig_tr(self)
    with torch.no_grad():
        for p_ in self.trunk.parameters():
            p_.add_(torch.full_like(p_, 1e-3))
    return out
V.VoProjBank.train = p_steps_plant
z = V.vo_proj_check()
rec("P-1", "the fit STEPS THE TRUNK (the read becomes a treatment)",
    not z["P-1:trunk_untouched"], f"trunk untouched = {z['P-1:trunk_untouched']}")
V.VoProjBank.train = orig_tr

# P3: the fit CONSUMES THE SHARED RNG STREAM — every draw downstream of it moves
orig_design = V.VoProjBank._design
def p_eats_rng(self, F, r, mu=None, sd=None, variant=None):
    import torch
    torch.rand(1)                       # outside the sandbox: the shared stream moves
    return orig_design(self, F, r, mu=mu, sd=sd, variant=variant)
V.VoProjBank._design = p_eats_rng
z = V.vo_proj_check()
rec("P-1", "the fit consumes the SHARED torch stream", not z["P-1:rng_unmoved"],
    f"shared streams unmoved = {z['P-1:rng_unmoved']}")
V.VoProjBank._design = orig_design

# P4: THE FEATURES ARE CACHED instead of recomputed against the moving trunk. This is the
# defect the recompute decision exists to avoid, and it is invisible to every inertness gate:
# a cached feature is more inert than a live one.
orig_feat = V.VoProjBank._features
def p_cached(self, x, r, blk, spn, mask=None, chunk=4096):
    key = (int(x.shape[0]), int(x.sum()), int(mask is None))
    cache = getattr(self, "_fcache", None)
    if cache is None:
        cache = self._fcache = {}
    if key not in cache:
        cache[key] = orig_feat(self, x, r, blk, spn, mask=mask, chunk=chunk)
    return cache[key]
V.VoProjBank._features = p_cached
z = V.vo_proj_check()
rec("P-3", "the features are CACHED, so a moving trunk cannot move the reading",
    not z["P-3:ok"],
    f"|dp| when the live trunk moved = {z['P-3:live_features_moved']:.3e} (must be > 0)")
V.VoProjBank._features = orig_feat

# P5: the "random" twin is SECRETLY THE LIVE TRUNK — the control is not a control
orig_init = V.VoProjBank.__init__
def p_twin_is_live(self, cfg, v_, length, s_, device, seed, trunk=None, tdim=None):
    orig_init(self, {**cfg, "vo_pj_trunk": "live"}, v_, length, s_, device, seed,
              trunk=trunk, tdim=tdim)
    self.which = "rand"                 # claims to be the twin, reads the live trunk
V.VoProjBank.__init__ = p_twin_is_live
z = V.vo_proj_check()
rec("P-2", "the 'random' twin is secretly the LIVE trunk",
    not (z["P-2:twin_is_another_object"] and z["P-3:ok"]),
    f"twin is another object = {z['P-2:twin_is_another_object']}  "
    f"twin |dp| under a live step = {z['P-3:twin_features_still']:.3e} (must be 0)")
V.VoProjBank.__init__ = orig_init

# P6: the twin is TRAINED — the frozen floor drifts into a second treatment
def p_twin_trained(self):
    import torch
    out = orig_tr(self)
    if self.which == "rand":
        with torch.no_grad():
            for p_ in self.trunk.parameters():
                p_.add_(torch.full_like(p_, 1e-3))
    return out
V.VoProjBank.train = p_twin_trained
z = V.vo_proj_check()
rec("P-2", "the random twin's trunk is TRAINED", not z["P-2:twin_frozen"],
    f"twin frozen = {z['P-2:twin_frozen']}")
V.VoProjBank.train = orig_tr

# P8 (P-5): THE DEFECT ITSELF — a constant-label buffer gets NO fit, so the grader silently
# goes dark and the probe channel is skipped on every cycle. This is what the preflight's M-1r
# caught on `al_pf_pj` ("no probe was ever drawn"), and it is the perturbation P-5 must fail on.
def p_no_degenerate(self):
    ys = self.y[(self.h >= self.hold) & (self.u[:, 0] < self.boot)]
    if int(ys.numel()) and float(ys.min()) == float(ys.max()):
        return None                      # the first version's behaviour, verbatim
    return orig_tr(self)
V.VoProjBank.train = p_no_degenerate
z = V.vo_proj_check()
rec("P-5", "a constant-label buffer gets NO fit, so the grader goes dark",
    not z["P-5:ok"],
    f"answered = {z['P-5:answered']}  ready = {z['P-5:ready']}  "
    f"degenerate refits = {z['P-5:n_degenerate']}")
V.VoProjBank.train = orig_tr

# P7: the readout is fit ONCE and never refit — disconnection, M-5's shape one organ over
def p_fit_once(self):
    if self.w is not None:
        self.n_refit += 1
        self.stat["n_refit"] = self.n_refit
        return self.stat.get("loss")
    return orig_tr(self)
V.VoProjBank.train = p_fit_once
z = V.vo_proj_check()
rec("P-4", "the readout is fit ONCE and never refit (DISCONNECTION)", not z["P-4:ok"],
    f"|dw| between refits = {z['P-4:weights_moved']:.3e} (must be > 0)")
V.VoProjBank.train = orig_tr
z = V.vo_proj_check()
rec("P-1..P-5", "restored (must PASS)", z["ok"],
    f"affine err={z['P-1:affine_err']:.2e} refits={z['P-4:n_refit']}")

# ============ [aliquot] P-2r / P-3r / P-4r: the run-level forms ======================= #
print("\n[aliquot] P-2r/P-3r/P-4r (the trunk's fingerprint over the run, off the arm file)")

def mk_proj_arm(which="live", sigs=(1.0, 1.1, 1.2), n_refit=3, lam=32.0,
                aucs=(0.8, 0.82, 0.85), lams=(1.0, 32.0, 1024.0),
                fit_base=0.3, n_degenerate=0):        # [soundboard]
    oms = [{"trunk_sig": (None if sg is None else float(sg)),
            "hold_auc": (None if au is None else float(au)), "n_rows": 1000,
            "fit_base": (None if fit_base is None else float(fit_base))}
           for sg, au in zip(sigs, aucs)]
    return {"vo_om_mode": "proj",
            "vo_om_state": {"which": which, "n_refit": int(n_refit), "lam": float(lam),
                            "lams": list(lams), "nfeat": 1737, "tdim": 96,
                            "n_degenerate": int(n_degenerate),
                            "sig0": float(sigs[0]), "refit_skipped": 0},
            "log": {"vo_om": oms}}

def runP(a, label, want_fail):
    try:
        d = V.vo_gate_proj_run(a, arm="pf")
        rec("P-2r/P-3r/P-4r", label, not want_fail,
            f"sig {d['sig_min']} .. {d['sig_max']}  refits {d['n_refit']}")
    except AssertionError as e:
        rec("P-2r/P-3r/P-4r", label, want_fail, str(e)[:110])

runP(mk_proj_arm("live"), "clean live arm (must PASS)", False)
runP(mk_proj_arm("rand", sigs=(2.0, 2.0, 2.0)), "clean rand arm (must PASS)", False)
runP(mk_proj_arm("rand", sigs=(2.0, 2.1, 2.2)), "the random twin's trunk MOVED over the run",
     True)
runP(mk_proj_arm("live", sigs=(1.0, 1.0, 1.0)),
     "the LIVE trunk never moved (the arm did not test a moving representation)", True)
runP(mk_proj_arm("live", n_refit=0), "the readout was never fit (VACUOUS)", True)
runP(mk_proj_arm("live", lam=7.0), "the ridge is not one of the stated grid", True)
runP(mk_proj_arm("live", aucs=(None, None, None)),
     "no held-out AUC was ever recorded", True)
# [soundboard] P-4r's strong form, CORRECTED (DESIGN §7.4): the claim is conditioned on the
# TRAINING slice having carried both classes, which is what `fit_base` records, and not on the
# held-out slice's. Both halves are shown: the defect trips, and the preflight-scale fact does not.
runP(mk_proj_arm("live", n_degenerate=3, fit_base=0.3),
     "the TRAINING slice had a contrast and every refit was still degenerate", True)
runP(mk_proj_arm("live", n_degenerate=3, fit_base=0.0),
     "every refit degenerate because the TRAINING slice was single-class (must PASS — the "
     "preflight-scale fact `sb_pf_so` tripped the inherited form on)", False)
runP(mk_proj_arm("live", n_degenerate=3, fit_base=None),
     "a file written before `fit_base` existed (must PASS — recorded, not asserted)", False)

# ============ [soundboard] S-1 .. S-5: the outcome error in the trunk's weights ======== #
print("\n[soundboard] S-1/S-1b/S-2/S-3/S-4/S-5 (the outcome error reaches the TRUNK)")
z = V.sb_shape_check()
rec("S-1..S-5", "unperturbed (must PASS)", z["ok"],
    f"|d sig| live={z['S-1:live:trunk_moved']:.3e} detached="
    f"{z['S-1:detached:trunk_moved']:.3e}  |dF|={z['S-2:max_abs_delta']:.3e}")

# S1: THE PERTURBATION THE LIVENESS GATE EXISTS FOR — the outcome gradient is DETACHED from the
# pooled state, so the head learns and the trunk does not. Nothing about the arm looks different:
# the term is computed, the BCE falls, the head moves, the log fills. This is the disconnection
# shape `voicing` V-6 was written for, one organ over, and it is what S-1 must go red on.
orig_term = V.VoShaper.term
def p_detached(self):
    import torch
    old_ = self.detach
    self.detach = True
    try:
        return orig_term(self)
    finally:
        self.detach = old_
V.VoShaper.term = p_detached
z = V.sb_shape_check()
rec("S-1", "the outcome gradient is DETACHED from the pooled state (DISCONNECTION)",
    not z["S-1:ok"],
    f"|d sig| on the LIVE arm = {z['S-1:live:trunk_moved']:.3e} (must be > 0)")
V.VoShaper.term = orig_term

# S2: the term is computed and never ADDED to the plant's loss — the other half of
# disconnection, and the one an inertness gate cannot see at all
def p_no_term(self):
    t_, st_ = orig_term(self)
    return (None if t_ is None else t_.detach() * 0.0), st_
V.VoShaper.term = p_no_term
z = V.sb_shape_check()
rec("S-1", "the term is ZEROED before it reaches the plant's loss", not z["S-1:ok"],
    f"|d sig| on the LIVE arm = {z['S-1:live:trunk_moved']:.3e} (must be > 0)")
V.VoShaper.term = orig_term

# S3: the shaping reads a DIFFERENT state from the one the projection reads — the mask dropped,
# so the shaping writes into a fully-unmasked input the plant has never seen and the grader
# never reads. Numerically small, structurally the wrong claim.
orig_in = V.sb_shape_input
def p_nomask(x, blk, spn, nb, s, mask, device):
    return orig_in(x, blk, spn, nb, s, False, device)
V.sb_shape_input = p_nomask
z = V.sb_shape_check()
rec("S-2", "the shaping reads the configuration UNMASKED (not the state the grader reads)",
    not z["S-2:ok"], f"|F - projection's F| = {z['S-2:max_abs_delta']:.3e} (must be 0)")
V.sb_shape_input = orig_in

# S4: the span pool is dropped, so the head reads the global pool alone — `duplex`'s finding was
# that most of the gain is context rather than candidate, which makes a silently context-only
# head exactly the confound the round must not have.
def p_meanonly(x, blk, spn, nb, s, mask, device):
    xb, wgt = orig_in(x, blk, spn, nb, s, mask, device)
    return xb, wgt * 0.0
V.sb_shape_input = p_meanonly
z = V.sb_shape_check()
rec("S-2", "the SPAN pool is dropped (the head reads the global pool alone)",
    not z["S-2:ok"], f"|F - projection's F| = {z['S-2:max_abs_delta']:.3e} (must be 0)")
V.sb_shape_input = orig_in

# S5: the shaping draws from the SHARED torch stream instead of its own
def p_shared_rng(self):
    import torch
    torch.rand(4)
    return orig_term(self)
V.VoShaper.term = p_shared_rng
z = V.sb_shape_check()
rec("S-3", "the shaping draws from the SHARED torch stream", not z["S-3:ok"],
    f"shared stream unmoved = {z['S-3:rng_unmoved']}")
V.VoShaper.term = orig_term

# S6: a "yield" shaper silently falls back to the VERDICT when the label column is empty — the
# shape of defect that would have made the currency arm a second copy of the verdict arm
orig_avail = V.VoShaper._avail
def p_fallback(self):
    import numpy as _np
    m = (self.bank.src == 0) & (self.bank.h >= self.bank.hold)
    return m.nonzero(as_tuple=True)[0].numpy()
V.VoShaper._avail = p_fallback
z = V.sb_shape_check()
rec("S-4", "a 'yield' shaper falls back to the VERDICT where the label is missing",
    not z["S-4:ok"],
    f"a yield target with no labels yields no term = "
    f"{z['S-4:yield_without_labels_is_none']} (must be True)")
V.VoShaper._avail = orig_avail

# S7: the yield label is computed against the RAW level-1 tuple instead of through the miner's
# own keying. On a quotiented miner (every arm in this lineage) that reads identically zero — a
# silent nil, and the exact shape `voicing` Q3's defect #8 had.
orig_keys = V.sb_miner_keys
def p_raw_keys(miner, feats):
    import numpy as _np
    f = _np.asarray(feats, _np.int64)
    return [tuple(int(z_) for z_ in f[i]) for i in range(f.shape[0])]
V.sb_miner_keys = p_raw_keys
z = V.sb_shape_check()
rec("S-5/S-6", "the yield label is keyed by the RAW tuple, not by the miner's own keys",
    not (z["S-5:ok"] and z["S-6:ok"]),
    f"label through a CLASS miner = {z['S-5:class_label']} and through the REAL "
    f"LearnedQuotient = {z['S-6:label']} (neither may be all zero)")
V.sb_miner_keys = orig_keys

# S9: the halves are keyed at the SPAN's level instead of one below it — an off-by-one in the
# level argument, which reads zero on every key because `observe` counted them one level down.
def p_wrong_level(miner, feats):
    import numpy as _np
    f = _np.asarray(feats, _np.int64)
    if not hasattr(miner, "quot"):
        return [tuple(int(z_) for z_ in f[i]) for i in range(f.shape[0])]
    half = miner.span // miner.s
    ids = [miner.quot.ids(f[:, i * half:(i + 1) * half], miner.level) for i in range(miner.s)]
    out = []
    for r in range(f.shape[0]):
        k = tuple(ids[i][r] for i in range(miner.s))
        out.append(None if any(z_ is None for z_ in k) else k)
    return out
V.sb_miner_keys = p_wrong_level
z = V.sb_shape_check()
rec("S-6b", "the halves are keyed at the SPAN's level, one above the miner's own",
    not z["S-6:ok"],
    f"under a MERGE the label reads {z['S-6b:label']} (want [1.0, 1.0]); without one the level "
    f"argument is inert, which is why the fixture merges first")
V.sb_miner_keys = orig_keys

# S8: the label ignores whether the piece SOLVED, so an unsolved configuration gets credit for
# chunks it never contributed
orig_lab = V.sb_yield_label
def p_no_solve(pf, miner, support, solved=None):
    return orig_lab(pf, miner, support, solved=None)
V.sb_yield_label = p_no_solve
z = V.sb_shape_check()
rec("S-5", "the yield label ignores the solve verdict", not z["S-5:ok"],
    f"label = {z['S-5:label']} (want [1.0, 0.125, 0.0, 0.0])")
V.sb_yield_label = orig_lab
z = V.sb_shape_check()
rec("S-1..S-5", "restored (must PASS)", z["ok"],
    f"|d sig| live={z['S-1:live:trunk_moved']:.3e} |dF|={z['S-2:max_abs_delta']:.3e}")

# ============ [soundboard] S-1r .. S-4r: the run-level forms ========================== #
print("\n[soundboard] S-1r/S-2r/S-3r/S-4r (off the arm file)")

def mk_sb_arm(on=True, target="solve", sigs=(1.0, 1.1, 1.2), steps=(0, 20, 40),
              bce=(0.7, 0.6, 0.55), label_mean=0.4, n_label=900, wm=True, dp=True,
              n_label_solved=180):                    # [soundboard]
    rows = []
    for sg, st_, b_ in zip(sigs, steps, bce):
        rows.append({"sig": float(sg),
                     "shape": ({"n_step": int(st_), "n_skip": 0, "n_rows": int(st_) * 64,
                                "bce": float(b_), "auc": 0.7, "base": 0.3,
                                "target": target} if on else None),
                     "wm": ({"infill_ce": 1.4, "infill_acc": 0.43, "parse_mask1": 0.64}
                            if wm else None)})
    return {"config": {"vo_sh": bool(on), "vo_sh_target": target, "vo_wm": bool(wm or dp)},
            "sb": {"on": bool(on), "target": (target if on else None), "infill": True,
                   "n_nograd": 0, "n_label": int(n_label), "label_mean": label_mean,
                   "label_pos": 300, "n_label_solved": int(n_label_solved), "dp_cycles": [1],
                   "wm_final": {"dp_parse": ({"2": {"acc": 0.4, "n": 100, "nodes": 2}}
                                             if dp else None)}},
            "log": {"sb": rows}}

def runS(a, label, want_fail):
    try:
        d = V.sb_gate_shape_run(a, arm="pf")
        rec("S-1r/S-2r/S-3r", label, not want_fail,
            f"steps {d['n_step']}  sig {d['sig_min']} .. {d['sig_max']}")
    except AssertionError as e:
        rec("S-1r/S-2r/S-3r", label, want_fail, str(e)[:110])

runS(mk_sb_arm(), "clean shaped arm (must PASS)", False)
runS(mk_sb_arm(on=False, sigs=(1.0, 1.0, 1.0), steps=(0, 0, 0)),
     "clean UNSHAPED twin (must PASS)", False)
runS(mk_sb_arm(steps=(0, 0, 0)), "the shaping never took a step (DISCONNECTION)", True)
runS(mk_sb_arm(sigs=(1.0, 1.0, 1.0)),
     "the plant never moved although the shaping ran", True)
runS(mk_sb_arm(target="yield", label_mean=0.0),
     "the yield currency is CONSTANT although labelled rows SOLVED (no gradient)", True)
# [soundboard] the other half, corrected (DESIGN §7.4): the currency is zero on an unsolved piece
# BY CONSTRUCTION, so a constant currency on an arm that never solved is not a defect and the
# assertion must not fire. This is the preflight-scale fact `sb_pf_yd` has.
runS(mk_sb_arm(target="yield", label_mean=0.0, n_label_solved=0),
     "the currency is constant because NOTHING SOLVED (must PASS — recorded, not asserted)",
     False)
runS(mk_sb_arm(target="yield", n_label=0),
     "the yield arm computed NO label (the head read the sentinel column)", True)
runS(mk_sb_arm(dp=False), "the nested DP parse never ran (no altitude instrument)", True)
runS({**mk_sb_arm(wm=False, dp=False), "config": {"vo_sh": True, "vo_sh_target": "solve",
                                                  "vo_wm": True}},
     "an arm naming `vo_wm` logged NO world-model diagnostic at all", True)
runS(mk_sb_arm(wm=False, dp=False),
     "an arm NOT naming `vo_wm` logs none (must PASS — the unshaped twin's shape)", False)

def runT(off, on, label, want_fail):
    try:
        d = V.sb_gate_shape_twin(off, on, arm_off="off", arm_on="on")["S-4r"]
        rec("S-4r", label, not want_fail,
            f"max|d sig| = {d['max_abs_delta']}  first differing cycle "
            f"{d['first_differing_cycle']}")
    except AssertionError as e:
        rec("S-4r", label, want_fail, str(e)[:110])

_off = mk_sb_arm(on=False, sigs=(1.0, 1.0, 1.0), steps=(0, 0, 0))
runT(_off, mk_sb_arm(sigs=(1.0, 1.1, 1.2)), "clean pair (must PASS)", False)
runT(_off, mk_sb_arm(sigs=(1.0, 1.0, 1.0)),
     "the shaped arm's plant is IDENTICAL to the unshaped twin's", True)
runT(_off, mk_sb_arm(sigs=(1.0, 1.1, 1.2), steps=(0, 0, 0)),
     "the shaped arm never took a shaping step (VACUOUS)", True)

bad = [r for r in res if not r[2]]
print(f"\nFINAL (Q3 + Q3b + sotto + aliquot + soundboard): {len(res)-len(bad)}/{len(res)} "
      f"perturbations behaved as required" + (f"  BLIND: {bad}" if bad else ""))
