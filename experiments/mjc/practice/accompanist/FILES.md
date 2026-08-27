# FILES — accompanist

**Up**: [README.md](README.md) · [../README.md](../README.md) (practice)

## Code files

None at this node — `accompanist/` is the super-writeup for one orchestrated conversation
(2026-08-26 → 08-27). The code lives in its child and in `offbook/`:

| where | what |
|---|---|
| [`presto/`](presto/FILES.md) | the fast-piece node: `piece.py`, `world.py`/`nets.py` (verbatim forks of offbook's), `tempo.py` (Phase A ladder), `delay_gate.py` (Phase B sweep with both delay operators), `analyze_tempo.py`, `analyze_delay.py` |
| [`../offbook/delay_gate.py`](../offbook/delay_gate.py) | Rounds 5–7b (`d1`–`d3b`): the content ladder, `build_auditioned`, `build_nested`, `--nest-adapt`, `--dual-obs` — additive flags whose defaults reproduce `d0` at 0.000e+00 |
| [`../offbook/world.py`](../offbook/world.py) | `obs_predict` (efference copy through the delay), ported verbatim from `presto/world.py`; G-F fork fidelity re-asserted at 0.000e+00 |
| [`../offbook/analyze_delay.py`](../offbook/analyze_delay.py) | reduction + figures for `d1`–`d3b`, incl. the legato fidelity block and the dual-incumbent verdicts |

## Children

| child | what it varies |
|---|---|
| [`presto/`](presto/FILES.md) | A piece meant to be playable only from memory: closed pentagon, 5 × 6-step (120 ms) segments, no patch, tempo × damping swept 3 × 3; the Δ sweep at the design point under both delay operators (naive, efference copy), nested library build with the FM adapting. |

## Related, living elsewhere

| node | what |
|---|---|
| [`../offbook/FILES.md`](../offbook/FILES.md) §Rounds 5–7b | `d1` content ladder · `d2` legato's nesting · `d3` the FM adapting · `d3b` the efference-copy incumbent — on offbook's own piece and gate |
| [`../acappella/SPEC.md`](../acappella/SPEC.md) | the model-free re-port this conversation licensed (spec only) |
