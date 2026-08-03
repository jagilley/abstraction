# shared_surface — File index

Complete file-by-file reference for this sub-experiment. Summarized in [README.md](README.md);
parent index at [`../FILES.md`](../FILES.md).

**This node adds no scripts of its own.** It is one new value of an existing variable, so the code
lives where the variable does (per [`STRUCTURE.md`](../../../../../../STRUCTURE.md) — code lives at
the lowest node that shares it):

| Where | What was added | Default |
|---|---|---|
| [`../../../../rhm_channels.py`](../../../../rhm_channels.py) | `share_rules_bottom` — the suffix splice (shared surface alphabet, independent deep composition), the mirror of `share_rules_top`. `make_layout` accepts `share_bottom` on any grammar channel and rejects setting both knobs on one channel | absent key ⇒ bit-identical |
| [`../../channel_env.py`](../../channel_env.py) | `make_spec(..., share_mode="top"\|"bottom")` — selects which end `struct_shares` splices | `"top"` |
| [`../geometry_check.py`](../geometry_check.py) | `--share-mode` on `geometry_check` / `share_probe`; **`level_cond_sibling_stats`** (G2c), the closed-form conditional-sibling certification that reads either splice direction exactly; a per-cell assertion that exactly *k* levels are shared at the end `share_mode` names | `"top"` |
| [`../aggregate.py`](../aggregate.py) | Results keyed on `share_mode` as well as `fm_arch` so the two DGPs are never pooled; **`cross_mode_check`**, which reads the full-sharing rung — where the two splices are the same DGP — as a measured run-to-run noise floor | files without the key ⇒ `"top"` |
| [`../../ladder.py`](../../ladder.py) | `--share-mode`; `shared_channels` recognises either knob | `"top"` |

Every addition defaults to the published behaviour, so every prior run on the parent node is
reproducible from the same scripts — confirmed by re-aggregating `geometry_g_s{1,2,3}` unchanged.

## Results

| Path | What |
|---|---|
| `../figures/geometry_sb_s{1,2,3}/results.json` | **G1–G4 across sharing depths 0–4 under `--share-mode bottom`** on the published block FM. Carries the G4 encoder-alignment table (§3 — chance → 0.563 centered at one shared table), the G2c closed-form certification, and the transfer curve whose cells lie inside the measured floor (§4). Results live in the **parent's** `figures/` because they are written by the parent's script to the parent's volume path |
| `../figures/geometry_smoke_sb/` | The `--quick` smoke that validated the wiring. `quick: true`, so `aggregate.py` skips it |

## Reading order

1. [README.md](README.md) §2–3 — the certification and the G4 result, which are what this node
   establishes.
2. [README.md](README.md) §4 — the transfer curve **and its measured floor**, which must be read
   together; the table alone is misleading.
3. [`../README.md`](../README.md) §4 — the parent's barrier (a)/(b) split, which §3 here resolves
   the first half of.
