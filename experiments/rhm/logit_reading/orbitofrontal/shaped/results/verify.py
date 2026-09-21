"""Touch every member of every fetched npz: `modal volume get` has silently truncated a
large npz and once returned one whose CRC was bad on a single member, which `np.load`
does not catch until the member is read."""
import glob
import sys

import numpy as np

root = sys.argv[1]
bad, n, mem = [], 0, 0
for p in sorted(glob.glob(f"{root}/**/*.npz", recursive=True)):
    n += 1
    try:
        Z = np.load(p)
        for k in Z.files:
            a = Z[k]
            mem += 1
            _ = float(np.asarray(a).reshape(-1)[:1].sum()) if a.size else 0.0
    except Exception as e:                                   # noqa: BLE001
        bad.append((p, repr(e)))
print(f"{n} npz, {mem} members touched, {len(bad)} bad")
for p, e in bad:
    print("BAD", p, e)
sys.exit(1 if bad else 0)
