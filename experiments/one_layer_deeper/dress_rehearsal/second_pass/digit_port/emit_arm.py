"""Write an arm-specialized copy of this node's `submission.py`.

Same mechanism as the parent node's `emit_arm.py`: the arms share every line except the
`ARM = "..."` constant, so the canonical file stays single and this rewrites that one line
into a target directory. Upstream's CLI and service require the file to be named
`submission.py`, hence a directory per arm.

    python emit_arm.py digit /tmp/digit_arms
    python emit_arm.py all /tmp/digit_arms
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ARMS = ("digit", "digit_learned", "digit_closure", "digit_carry")
PATTERN = re.compile(r'^ARM = "(?:' + "|".join(ARMS) + r')"$', re.MULTILINE)


def emit(arm: str, source: Path, out_dir: Path) -> Path:
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {ARMS}")
    text = source.read_text(encoding="utf-8")
    patched, count = PATTERN.subn(f'ARM = "{arm}"', text)
    if count != 1:
        raise SystemExit(f"expected exactly one ARM line in {source}, found {count}")
    target = out_dir / arm
    target.mkdir(parents=True, exist_ok=True)
    path = target / "submission.py"
    path.write_text(patched, encoding="utf-8")
    payload = patched.encode("utf-8")
    if len(payload) > 256 * 1024:
        raise SystemExit(f"{path} is {len(payload)} bytes, over the 256 KiB limit")
    print(f"{path}  sha256={hashlib.sha256(payload).hexdigest()}  bytes={len(payload)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=[*ARMS, "all"])
    parser.add_argument("out_dir")
    parser.add_argument("--source", default=str(Path(__file__).with_name("submission.py")))
    args = parser.parse_args()
    for arm in ARMS if args.arm == "all" else (args.arm,):
        emit(arm, Path(args.source), Path(args.out_dir))


if __name__ == "__main__":
    main()
