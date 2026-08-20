"""Rung B1 driver: walk reasoners through open-loop sequential sessions over measured trajectories.

Reasoner = Claude via headless `claude -p`, model pinned explicitly.  Every full prompt and every
full response is written to JSON; nothing is summarised at call time.

    python3 rhm/practice/teacher_slot/verbal/run_sessions.py --dry-run          # render only
    python3 rhm/practice/teacher_slot/verbal/run_sessions.py --tag b1 --cells pre_task,pre_full,dead_ctrl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from rhm.practice.teacher_slot.verbal import vignette as V          # noqa: E402
from rhm.practice.teacher_slot.verbal.records import CELLS, Session  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
NEUTRAL_CWD = os.environ.get("B1_CWD", "/tmp/b1_reasoner_cwd")

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


# ------------------------------------------------------------------ the reasoner call
def call_claude(prompt: str, model: str, timeout: int = 420) -> dict:
    os.makedirs(NEUTRAL_CWD, exist_ok=True)
    cmd = [
        "claude", "-p",
        "--model", model,
        "--system-prompt", V.SYSTEM_PROMPT,
        "--tools", "",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--no-session-persistence",
        "--output-format", "json",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, input=prompt, cwd=NEUTRAL_CWD, capture_output=True,
                       text=True, timeout=timeout)
    dt = time.time() - t0
    rec = {"cmd": cmd, "returncode": p.returncode, "seconds": round(dt, 2),
           "stderr": p.stderr[-2000:] if p.stderr else ""}
    try:
        j = json.loads(p.stdout)
        rec["text"] = j.get("result", "")
        rec["model_reported"] = (j.get("modelUsage") and list(j["modelUsage"].keys())) or None
        rec["session_id"] = j.get("session_id")
        rec["usage"] = j.get("usage")
        rec["raw_json_keys"] = sorted(j.keys())
    except Exception as e:                                    # noqa: BLE001
        rec["text"] = p.stdout
        rec["parse_error"] = repr(e)
    return rec


CHOICE_RE = re.compile(r"^\s*CHOICE\s*:\s*(.*)$", re.M | re.I)
FIELD_RE = {
    "reliance": re.compile(r"^\s*RELIANCE\s*:\s*(.*?)(?=^\s*(?:STABILITY|CHOICE|REASONS)\s*:|\Z)",
                           re.M | re.I | re.S),
    "stability": re.compile(r"^\s*STABILITY\s*:\s*(.*?)(?=^\s*(?:RELIANCE|CHOICE|REASONS)\s*:|\Z)",
                            re.M | re.I | re.S),
    "reasons": re.compile(r"^\s*REASONS\s*:\s*(.*?)(?=^\s*(?:RELIANCE|STABILITY|CHOICE)\s*:|\Z)",
                          re.M | re.I | re.S),
}


def parse(text: str) -> dict:
    out = {k: None for k in ("reliance", "stability", "reasons")}
    for k, rx in FIELD_RE.items():
        m = rx.search(text)
        if m:
            out[k] = " ".join(m.group(1).split())
    ch = None
    m = CHOICE_RE.search(text)
    if m:
        blob = m.group(1).upper()
        has_l, has_r = "LEAVE_T" in blob, "REPLACE_T" in blob
        if has_l and not has_r:
            ch = "LEAVE_T"
        elif has_r and not has_l:
            ch = "REPLACE_T"
    out["choice"] = ch
    return out


# ------------------------------------------------------------------ one session
OPT_ORDERS = ["LR", "RL", "LR", "RL", "LR"]


def run_session(cell_name: str, sample: int, model: str, splice: str, dry: bool) -> dict:
    cell = CELLS[cell_name]
    opt_order = OPT_ORDERS[sample % len(OPT_ORDERS)]
    s = Session(cell, splice=splice)
    turns = []
    while not s.done:
        prompt = V.render(cell, s.rows, s.choices, s.step, opt_order)
        leaks = V.check_blinding(prompt)
        asym = V.check_symmetry(prompt)
        if leaks or asym:
            raise RuntimeError(f"BLINDING/SYMMETRY FAILURE at {cell_name} s{sample} "
                               f"step {s.step}: {leaks} {asym}")
        row = s.rows[-1]
        turn = {"step": s.step, "prompt": prompt,
                "row": {"step": row.step, "underlying": row.underlying, "src": row.src,
                        "in_stream": row.in_stream, "nllA": row.nllA, "nllB": row.nllB,
                        "nllA_star": row.nllA_star, "nllB_star": row.nllB_star,
                        "dec": row.dec, "dec_step": row.dec_step},
                "removal_arm_m": s.rem_m, "removed_at": s.removed_at,
                "attempts": []}
        if dry:
            choice = "LEAVE_T"
            turn["parsed"] = {"choice": choice, "reliance": None, "stability": None,
                              "reasons": None}
        else:
            choice = None
            for attempt in range(3):
                rec = call_claude(prompt, model)
                pr = parse(rec.get("text", ""))
                turn["attempts"].append({"call": rec, "parsed": pr})
                if pr["choice"]:
                    choice = pr["choice"]
                    turn["parsed"] = pr
                    break
                log(f"    [{cell_name} s{sample} step {s.step}] unparseable "
                    f"(attempt {attempt+1}); retrying")
            if choice is None:
                turn["parsed"] = {"choice": None}
                raise RuntimeError(f"no parseable CHOICE at {cell_name} s{sample} step {s.step}")
        turns.append(turn)
        log(f"  [{cell_name} s{sample}] step {s.step:>6} -> {choice}")
        s.record_choice(choice)
    return {"cell": cell_name, "sample": sample, "model": model, "splice": splice,
            "opt_order": opt_order, "join_at": cell.join_at, "grid": cell.grid,
            "info": cell.info, "t_fact": cell.t_fact,
            "kept_arm": f"{cell.kept_tag}/{cell.kept_arm}",
            "removal_arms": [f"{t}/{n}@{m}" for m, t, n in cell.removal_arms],
            "choices": s.choices, "turns": turns,
            "final_rows": [{"step": r.step, "underlying": r.underlying, "in_stream": r.in_stream,
                            "src": r.src, "nllA": r.nllA, "nllB": r.nllB,
                            "nllA_star": r.nllA_star, "nllB_star": r.nllB_star,
                            "dec": r.dec, "dec_step": r.dec_step} for r in s.rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1")
    ap.add_argument("--cells", default="pre_task,pre_full,dead_ctrl")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--late-samples", type=int, default=2)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--splice", default="elapsed", choices=["elapsed", "anchored"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for c in [x for x in a.cells.split(",") if x]:
        n = a.late_samples if c == "late_task" else a.samples
        jobs += [(c, i) for i in range(n)]
    log(f"[b1] {len(jobs)} sessions, model={a.model}, splice={a.splice}, "
        f"workers={a.workers}, dry={a.dry_run}")

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_session, c, i, a.model, a.splice, a.dry_run): (c, i)
                for c, i in jobs}
        for f in futs:
            pass
        for f, key in futs.items():
            try:
                results.append(f.result())
            except Exception as e:                             # noqa: BLE001
                log(f"[b1] ERROR {key}: {e!r}")
                errors.append({"job": key, "error": repr(e)})

    path = os.path.join(OUT, f"sessions_{a.tag}.json")
    with open(path, "w") as fh:
        json.dump({"tag": a.tag, "model_arg": a.model, "splice": a.splice,
                   "samples": a.samples, "system_prompt": V.SYSTEM_PROMPT,
                   "sessions": results, "errors": errors}, fh, indent=1)
    log(f"[b1] wrote {path}  ({len(results)} sessions, {len(errors)} errors)")
    log("App completed")


if __name__ == "__main__":
    main()
