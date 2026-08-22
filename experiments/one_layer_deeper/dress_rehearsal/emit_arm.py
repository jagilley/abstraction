"""Write an arm-specialized copy of `submission.py`.

The two arms share every line except the `ARM = "..."` constant, so the canonical file is
kept single and this rewrites that one line into a target directory. Upstream's CLI and
service require the file to be named `submission.py`, hence a directory per arm.

    python emit_arm.py control /tmp/arms
    python emit_arm.py closure /tmp/arms
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ARMS = ("control", "closure")
PATTERN = re.compile(r'^ARM = "(?:control|closure)"$', re.MULTILINE)


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
    size = len(patched.encode("utf-8"))
    if size > 256 * 1024:
        raise SystemExit(f"{path} is {size} bytes, over the 256 KiB submission limit")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=[*ARMS, "all"])
    parser.add_argument("out_dir")
    parser.add_argument("--source", default=str(Path(__file__).with_name("submission.py")))
    args = parser.parse_args()
    arms = ARMS if args.arm == "all" else (args.arm,)
    for arm in arms:
        print(emit(arm, Path(args.source), Path(args.out_dir)))


if __name__ == "__main__":
    main()
