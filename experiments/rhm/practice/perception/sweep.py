"""The CPU coordinator. Fans the node's four GPU stages across containers in two waves,
so the wall clock is the longest stage plus the dependent ones rather than their sum.

  wave 1 (parallel)   record_run           re-run conductor's `outer_yield` with the
                                           recorder (~35-40 min)
                      train_reader corpus  the reader on the practice grammar
  wave 2              train_reader own     the reader on the learner's own productions
                                           (needs wave 1's pieces)
  wave 3 (parallel)   evaluate corpus / evaluate own

Run (from experiments/):
  modal run --detach -m rhm.practice.perception.sweep::sweep --tag pc0
"""

import json
import time

from rhm.shared import DATA_DIR, NumpyEncoder, volume
from rhm.practice.perception.common import REMOTE, app
from rhm.practice.perception.record import check_fidelity, record_run
from rhm.practice.perception.evaluate import evaluate
from rhm.practice.perception.reader import train_reader


@app.function(volumes={DATA_DIR: volume}, timeout=8 * 3600, memory=2048, cpu=1.0)
def sweep(tag: str = "pc0", corpus_steps: int = 32000, own_steps: int = 16000,
          window: int = 128, cd_arm: str = "outer_yield", skip_record: bool = False,
          own_ckpts: str = "0,250,1000,2000,4000,8000,16000",
          corpus_ckpts: str = "0,250,1000,2000,4000,8000,16000,32000"):
    t0 = time.time()
    out = {"tag": tag, "stages": {}}

    rec = None if skip_record else record_run.spawn(tag=tag, arm=cd_arm)
    corp = train_reader.spawn(tag=tag, arm="corpus", window=window,
                              steps=corpus_steps, ckpt_steps=corpus_ckpts)
    if rec is not None:
        out["stages"]["record"] = rec.get()
        print(f"[sweep] record done at {time.time() - t0:.0f}s", flush=True)
    fid = check_fidelity.spawn(tag=tag, arm=cd_arm)
    own = train_reader.spawn(tag=tag, arm="own", window=window, steps=own_steps,
                             ckpt_steps=own_ckpts)
    for k, c in (("reader_corpus", corp), ("fidelity", fid), ("reader_own", own)):
        out["stages"][k] = c.get()
        print(f"[sweep] {k} done at {time.time() - t0:.0f}s", flush=True)

    ev = {a: evaluate.spawn(tag=tag, arm=a, cd_arm=cd_arm) for a in ("corpus", "own")}
    for k, c in ev.items():
        out["stages"][f"eval_{k}"] = c.get()
        print(f"[sweep] eval_{k} done at {time.time() - t0:.0f}s", flush=True)

    out["elapsed_s"] = time.time() - t0
    with open(f"{DATA_DIR}/{REMOTE}/{tag}/sweep.json", "w") as f:
        json.dump(out, f, indent=1, cls=NumpyEncoder)
    volume.commit()
    print(f"[sweep] DONE {out['elapsed_s']:.0f}s", flush=True)
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=3 * 3600, memory=2048, cpu=1.0)
def smoke(tag: str = "pcsmoke", window: int = 128):
    """Mechanics only, at toy sizes: a 3-cycle practice run, a 300-step reader, the whole
    evaluation path. Nothing here is a measurement."""
    t0 = time.time()
    out = {"tag": tag, "stages": {}}
    out["stages"]["record"] = record_run.remote(tag=tag, arm="outer_yield", quick=True)
    out["stages"]["reader_corpus"] = train_reader.remote(
        tag=tag, arm="corpus", window=window, steps=300, ckpt_steps="0,300",
        chunk_seqs=40000, log_interval=100)
    out["stages"]["reader_own"] = train_reader.remote(
        tag=tag, arm="own", window=window, steps=300, ckpt_steps="0,300", log_interval=100)
    for a in ("corpus", "own"):
        out["stages"][f"eval_{a}"] = evaluate.remote(tag=tag, arm=a, n_corpus=256)
    out["elapsed_s"] = time.time() - t0
    print(f"[smoke] DONE {out['elapsed_s']:.0f}s", flush=True)
    return out
