# native — File & Child Index

Summarized in [README.md](README.md) (the single writeup for the node); [SPEC.md](SPEC.md) is the
record of what was asked. Children carry machinery/decision `FILES.md` records, no READMEs by
policy (the `typed_gaps` convention).

## Children

| Child | What |
|---|---|
| [`prop/`](prop/FILES.md) | **Port 1 — routing.** `prop.py` (verbatim `ratchet.py` fork + the proposal beam), `prop_net.py` (head, stable-sort top-k selection, ragged expansion, z-cache pricing, exploration), `analyze_prop.py`, `launch_detached.py`. Gates P-1…P-7; run `np_s0` (12 arms). |
| [`span/`](span/FILES.md) | **Port 2 — corridor.** `span.py` (verbatim fork + the span-trained plant loop), `span_net.py` (trunk-sharing span head, slot conditioning, per-macro parity gate, Plain/Span executors), `analyze_span.py`, `launch_detached.py`. Gates S-1…S-6; run `sp_s0` (5 arms). |
| [`full/`](full/FILES.md) | **Composition + readouts.** `full.py` (forks `prop.py`, imports `span_net.py`; end-of-run ablation battery; poison pair), `analyze_full.py`, `launch_detached.py`. Gates C-1…C-5 (C-5 added after a smoke-caught shadowing bug — the calibration story is in the child's `FILES.md`); run `nf_s0` (7 arms). |

## Code files

No code at this level; every script lives in a child and each child's `FILES.md` carries per-file
purposes and the design decisions on the record (pricing accounting, parity operationalization,
exploration, ablation-battery policies). `full/` imports from `prop/` and `span/`; nothing here
modifies `../ratchet/` or `../teacher_slot/`.
