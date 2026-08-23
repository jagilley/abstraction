"""Persist one hosted run into `results/hosted/<tier>_<dataset>_<arm>[_vN]/`.

Same layout as the parent node's: `status.json` (the full `one-layer status --json`
payload, seven rungs on both depth profiles), `metrics.jsonl` (the `log_every` training
curve), and `submission.json` (submission id, the SHA-256 of the exact file submitted, and
a change note).

    python save_run.py medium m6 digit --id <submission-id> \
        --file /tmp/digit_arms/digit/submission.py --note "first per-digit arm"
    python save_run.py medium m6 digit --id <id> --file ... --note ... --tag medium_m6_digit_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOSTED = ROOT / "results" / "hosted"


def _status(submission_id: str) -> dict:
    raw = subprocess.run(
        ["one-layer", "status", submission_id, "--json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    start = raw.index("{")
    return json.loads(raw[start:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier")
    parser.add_argument("dataset")
    parser.add_argument("arm")
    parser.add_argument("--id", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    tag = args.tag or f"{args.tier}_{args.dataset}_{args.arm}"
    out = HOSTED / tag
    out.mkdir(parents=True, exist_ok=True)

    status = _status(args.id)
    (out / "status.json").write_text(json.dumps(status, indent=2))

    subprocess.run(
        ["one-layer", "metrics", args.id, "--output", str(out / "metrics.jsonl")],
        check=False,
        capture_output=True,
    )

    payload = Path(args.file).read_bytes()
    (out / "submission.json").write_text(
        json.dumps(
            {
                "submission_id": args.id,
                "tier": args.tier,
                "dataset": args.dataset,
                "arm": args.arm,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "note": args.note,
            },
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()
