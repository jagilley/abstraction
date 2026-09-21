"""Touch every member of every fetched .npz.  `modal volume get` has silently truncated
one, and once returned one with a bad CRC on a single member that `np.load` did not catch,
so opening the file is not enough -- every array has to be materialised."""
import glob
import sys

import numpy as np

root = sys.argv[1]
bad = 0
for p in sorted(glob.glob(f"{root}/**/*.npz", recursive=True)):
    try:
        z = np.load(p)
        n = 0
        for k in z.files:
            a = z[k]
            n += int(np.asarray(a).size)
        print(f"ok   {p}  {len(z.files)} members, {n:,} values")
    except Exception as e:
        bad += 1
        print(f"BAD  {p}  {e!r}")
print(f"{bad} bad file(s)")
sys.exit(1 if bad else 0)
