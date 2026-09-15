"""[sotto] (`voicing`'s, copied and retargeted at `rhm_practice_sotto`)

[enharmonic] Fetch a tag from the volume and rewrite it as the COMPACT local mirror.

WHY. The repo checks in a per-arm `results.json` mirror by convention, and this node's arm
files are 205-345 MB against ~7.5 MB in `tu_s0`. Measured on `en_smoke`, the cause is one
object: the entry recorder's PROBE-phase vector is sized by the executed macro's own table, the
audition executes the TRUE table at every level up to `max_macro_level`, and at this node's
`maxl = 5` that table has 262,144 rows. Logged once per cycle it is 60.7 MB of the 61.0 MB an
arm file weighs compact; the slot-resolved record and the quotient log together are 0.10 MB.
`enharmonic_run`'s `entry_rec_cap` fixes it going forward (probe phase only — the beam phase is
never capped, because its vectors are over committed rows and are what every reduction reads).
This script is for the runs launched before that, `en_s0` included, and it is cheap insurance
afterwards.

WHAT IT WRITES, per arm, beside the fetched file:
    results.json      the whole record MINUS `log["entry"]`, compact (no indent)
    entry.json.gz     `log["entry"]` alone, gzipped
    entry_beam.json   the BEAM-phase record alone, compact and ungzipped — the object the
                      reduction actually reads, small enough to grep

Nothing is deleted before both companions are written, and the original is kept as
`results.raw.json` unless `--replace` is passed.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/fetch_compact.py --tag en_s0 --fetch --replace
    python3 rhm/practice/voicing/sotto_voce/fetch_compact.py --tag en_smoke --replace   # no re-fetch
"""

import argparse
import gzip
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_sotto"


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def compact_arm(path, replace=False):
    """Split one arm file. Returns a row of the before/after sizes."""
    d = os.path.dirname(path)
    n0 = os.path.getsize(path)
    with open(path) as f:
        r = json.load(f)
    entry = (r.get("log") or {}).pop("entry", None)
    beam = None
    if entry is not None:
        beam = [{"cycle": i, "beam": (e.get("hist") or {}).get("beam"),
                 "true_mask": e.get("true_mask")} for i, e in enumerate(entry)]
        with gzip.open(os.path.join(d, "entry.json.gz"), "wt") as f:
            json.dump(entry, f, separators=(",", ":"))
        with open(os.path.join(d, "entry_beam.json"), "w") as f:
            json.dump(beam, f, separators=(",", ":"))
    tmp = path + ".compact"
    with open(tmp, "w") as f:
        json.dump(r, f, separators=(",", ":"))
    if replace:
        os.replace(tmp, path)
    else:
        os.replace(path, os.path.join(d, "results.raw.json"))
        os.replace(tmp, path)
    return {"arm": os.path.basename(d), "before_mb": n0 / 1e6,
            "after_mb": os.path.getsize(path) / 1e6,
            "entry_gz_mb": (os.path.getsize(os.path.join(d, "entry.json.gz")) / 1e6
                            if entry is not None else 0.0),
            "beam_mb": (os.path.getsize(os.path.join(d, "entry_beam.json")) / 1e6
                        if entry is not None else 0.0),
            "n_cycles": len(entry or [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="en_s0")
    ap.add_argument("--fetch", action="store_true")
    # [voicing Q3b] USE `--replace`. Without it `compact_arm` keeps the pre-split original as
    # `results.raw.json`, which is how `vo_s3`'s mirror came through at 459 MB against
    # `vo_s1`/`vo_s2`'s 136/147 MB — not a regression in this script and not a new file from the
    # run, just the flag. The raw is fully reconstructible from `results.json` +
    # `entry.json.gz` (verified field by field on `vo_s3/voi3_comp`: same top-level keys, every
    # surviving log series equal, `events` and `config` equal, and `entry` recovered exactly),
    # so keeping it buys nothing and costs ~345 MB a tag.
    ap.add_argument("--replace", action="store_true",
                    help="drop the raw file instead of keeping it as results.raw.json "
                         "(THE IDIOM: vo_s1/vo_s2 were fetched this way)")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    root = os.path.join(FIG, a.tag)
    rows = []
    for arm in sorted(os.listdir(root)):
        p = os.path.join(root, arm, "results.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            head = f.read(4096)
        if '"entry"' not in head and os.path.getsize(p) < 50e6:
            print(f"  [skip] {arm}: already compact")
            continue
        rows.append(compact_arm(p, replace=a.replace))
        r = rows[-1]
        print(f"  {r['arm']:>16}: {r['before_mb']:>7.1f} MB -> {r['after_mb']:>6.1f} MB "
              f"+ entry.json.gz {r['entry_gz_mb']:>5.1f} MB "
              f"+ entry_beam.json {r['beam_mb']:>5.2f} MB   ({r['n_cycles']} cycles)")
    if rows:
        print(f"\n  total {sum(r['before_mb'] for r in rows):.0f} MB -> "
              f"{sum(r['after_mb'] + r['entry_gz_mb'] + r['beam_mb'] for r in rows):.0f} MB")


if __name__ == "__main__":
    main()
