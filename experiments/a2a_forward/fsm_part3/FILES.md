# FILES — `fsm_part3/`

Complete index for part 3 of the forward-self-model program. Standing context and all results
live in [README.md](README.md); the parent index is [../FILES.md](../FILES.md). The node spans
both of paper 2's substrates and lives here, under the program's home, rather than split across
the two harness folders it forks.

## Code files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker so `a2a_forward.fsm_part3.*` imports under Modal's `add_local_python_source("a2a_forward")`. The grammar child imports the `rhm` package and runs under the rhm image; it is launched by path, not by module. |

## Children

| Folder | Summary |
|---|---|
| [`authorship/`](authorship/FILES.md) ([design](authorship/DESIGN.md)) | **Can M tell which tokens it wrote?** M's temperature-1 samples spliced with corpus text (`mix` and splice-free `whole` sets), per-position "mine" target, paper 2's battery plus a likelihood-only observer and a sequence-level self-reporter. Self 0.533 vs likelihood observer 0.757 on the positions where IMPL reads +0.270; `O_act` at the self-report; the small self-signal dies under `shuffle_p` and not `shuffle_r`. Phase 2 (a bolt-on efference copy) designed, not built. 1.4 GPU-h. |
| [`dispositional_language/`](dispositional_language/FILES.md) ([design](dispositional_language/DESIGN.md)) | **Does the privilege extend to how M's computation is changing? (language)** Checkpointed OL wake, fresh instruments per checkpoint, the 2×2 of {implementation, behavior} × {prospective, retrospective}, three instrument readings of reorganization, historian observer, strict held-out-checkpoint scoring; `stratified.py` re-fits the heads and observers and cuts by syntactic category, entropy quartile, position and occurrent \|r\| quartile. Retrospective implementation advantage +0.026 (rank +0.071) against the best public observer, concentrating in the top \|r\| quartile (+0.067); prospective is instrument variation; behavior public in every stratum. 5.1 GPU-h. |
| [`dispositional_grammar/`](dispositional_grammar/FILES.md) ([design](dispositional_grammar/DESIGN.md)) | **The same on the grammar**, CL and OL arms, horizons `k=1` and `k=3`, geometric checkpoints, everything read within hierarchy level, `O_hist` and `O_act` ceilings per level, fixed-instrument reorganization rows and the fresh-instrument floor by level. On OL the implementation rows carry +0.10 to +0.29 at L3–L5 in both directions against every ceiling while behavior is flat at every level; L0 (half the positions) hides it in the pool; the CL arm shrinks it. Follow-ups (fresh reading through the cells; continuous within level) in `figures/ol_followups_reduction.txt`. 8 GPU-h. |

## Related, outside this folder

| Path | Relation |
|---|---|
| [`../confabulation/`](../confabulation/README.md) | Paper 2's language battery; `authorship/` and `dispositional_language/` fork it and import its helpers unchanged |
| [`../../rhm/confabulation/`](../../rhm/confabulation/README.md) | Paper 2's grammar battery; `dispositional_grammar/` forks it |
| [`../../mjc/committee_head/README.md`](../../mjc/committee_head/README.md) | The target-learnability lesson the autocorrelation gate comes from |
| [`../../rhm/residual_decomposition/trajectory/README.md`](../../rhm/residual_decomposition/trajectory/README.md) | The residual's reorganization along training, the raw material of the dispositional targets |
| `../../../papers/forward_self_models_paper2.md`[^private] | The definitions every number here is read against |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
