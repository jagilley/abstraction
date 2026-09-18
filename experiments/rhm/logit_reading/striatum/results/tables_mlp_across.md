# striatum -- the across-level contrast on the MLP critic (re-read, 2026-09-17)

Pure re-read of banked `stepNNNNNN_striatum_<tag>.npz` artefacts: no model is run and nothing is refit. `task.py::across_level` is called on the identical held-out rows for the trained **linear** critic (the banked `across_level`), the **MLP** critic on the same block (never read this way before) and the **clean-only** critic (the banked `across_level_clean`).

Reproduction:

```bash
cd experiments   # MODAL_PROFILE=chromatic
D=/v16_s2_L6_m4_distinct/logit_reading
modal volume get rhm-scaling-data $D/traj_a1_s42 <dir>/traj_a1_s42
modal volume get rhm-scaling-data $D/traj_eps01_s42 <dir>/traj_eps01_s42
python -m rhm.logit_reading.striatum.analyze <dir>/traj_a1_s42 --mlp-across \
    --extra-dirs <dir>/traj_eps01_s42 --tags a1,swap65k \
    --out rhm/logit_reading/striatum/results
```

**Reading.** At a fixed `(window, anchor, a)` the state is identical across query levels, so the contrast between the levels whose answer the edit changed and the levels whose answer it did not is matched on everything a state-only readout can see. Levels are ranked within level (`l <= 4`); `win rate` is the fraction of rows whose mean rank over consequential levels beats its mean rank over inconsequential levels (0.5 = null, ties count 0.5). `scalar rank` is the window's mean rank over all valid levels -- the quantity a scalar readout could express.

**Two stratifications.** The pooled row above ranks within level but pools across edit widths `j` and detection depths `k*`, both of which move the SCALE of the revision, so a wide or deep edit's large revisions are ranked against narrow ones. The per-`j` and per-`(j, k*)` tables re-rank inside the cell, which removes that between-cell scale difference; the `n`-weighted row is the controlled version of the same contrast. Cells below a 30-row floor are dropped, so `coverage` reports the share of the pooled row set the surviving cells retain.

## Reproduction gate: banked JSON vs this re-read

Every `linear` and `clean` win rate below is recomputed from the `.npz` and compared with the banked `tables.<anchor>.across_level{,_clean}` cell in the same run's JSON (`n` must match too). The `.npz` stores the revisions as `float32` while the banked JSON ranked them in `float64`, so near-ties can flip a rank in a small stratum; the tolerance is 0.005 and the largest deviation observed here is **0.0028**.

| traj           | venue   | step  | anchor | cell         | banked | re-read |    |
|----------------|---------|-------|--------|--------------|--------|---------|----|
| traj_a1_s42    | a1      | 8000  | fd     | linear a0    | 0.522  | 0.522   | OK |
| traj_a1_s42    | a1      | 8000  | fd     | linear a1    | 0.500  | 0.500   | OK |
| traj_a1_s42    | a1      | 8000  | fd     | clean a0     | 0.525  | 0.525   | OK |
| traj_a1_s42    | a1      | 8000  | fd     | clean a1     | 0.499  | 0.499   | OK |
| traj_a1_s42    | a1      | 8000  | fd     | linear j1 a0 | 0.582  | 0.582   | OK |
| traj_a1_s42    | a1      | 8000  | fd     | linear j2 a0 | 0.500  | 0.500   | OK |
| traj_a1_s42    | a1      | 8000  | fd     | linear j3 a0 | 0.495  | 0.495   | OK |
| traj_a1_s42    | a1      | 8000  | fd     | linear j4 a0 | 0.543  | 0.543   | OK |
| traj_a1_s42    | a1      | 8000  | tv     | linear a0    | 0.482  | 0.482   | OK |
| traj_a1_s42    | a1      | 8000  | tv     | linear a1    | 0.503  | 0.503   | OK |
| traj_a1_s42    | a1      | 8000  | tv     | clean a0     | 0.473  | 0.473   | OK |
| traj_a1_s42    | a1      | 8000  | tv     | clean a1     | 0.500  | 0.500   | OK |
| traj_a1_s42    | a1      | 8000  | tv     | linear j1 a0 | 0.479  | 0.479   | OK |
| traj_a1_s42    | a1      | 8000  | tv     | linear j2 a0 | 0.466  | 0.466   | OK |
| traj_a1_s42    | a1      | 8000  | tv     | linear j3 a0 | 0.573  | 0.573   | OK |
| traj_a1_s42    | a1      | 24000 | fd     | linear a0    | 0.527  | 0.527   | OK |
| traj_a1_s42    | a1      | 24000 | fd     | linear a1    | 0.509  | 0.509   | OK |
| traj_a1_s42    | a1      | 24000 | fd     | clean a0     | 0.536  | 0.536   | OK |
| traj_a1_s42    | a1      | 24000 | fd     | clean a1     | 0.516  | 0.516   | OK |
| traj_a1_s42    | a1      | 24000 | fd     | linear j1 a0 | 0.542  | 0.542   | OK |
| traj_a1_s42    | a1      | 24000 | fd     | linear j2 a0 | 0.469  | 0.469   | OK |
| traj_a1_s42    | a1      | 24000 | fd     | linear j3 a0 | 0.491  | 0.491   | OK |
| traj_a1_s42    | a1      | 24000 | fd     | linear j4 a0 | 0.553  | 0.553   | OK |
| traj_a1_s42    | a1      | 24000 | tv     | linear a0    | 0.424  | 0.424   | OK |
| traj_a1_s42    | a1      | 24000 | tv     | linear a1    | 0.500  | 0.500   | OK |
| traj_a1_s42    | a1      | 24000 | tv     | clean a0     | 0.422  | 0.422   | OK |
| traj_a1_s42    | a1      | 24000 | tv     | clean a1     | 0.516  | 0.516   | OK |
| traj_a1_s42    | a1      | 24000 | tv     | linear j1 a0 | 0.473  | 0.473   | OK |
| traj_a1_s42    | a1      | 24000 | tv     | linear j2 a0 | 0.480  | 0.477   | OK |
| traj_a1_s42    | a1      | 24000 | tv     | linear j3 a0 | 0.406  | 0.406   | OK |
| traj_a1_s42    | a1      | 64000 | fd     | linear a0    | 0.513  | 0.513   | OK |
| traj_a1_s42    | a1      | 64000 | fd     | linear a1    | 0.506  | 0.506   | OK |
| traj_a1_s42    | a1      | 64000 | fd     | clean a0     | 0.514  | 0.514   | OK |
| traj_a1_s42    | a1      | 64000 | fd     | clean a1     | 0.516  | 0.516   | OK |
| traj_a1_s42    | a1      | 64000 | fd     | linear j1 a0 | 0.491  | 0.491   | OK |
| traj_a1_s42    | a1      | 64000 | fd     | linear j2 a0 | 0.489  | 0.489   | OK |
| traj_a1_s42    | a1      | 64000 | fd     | linear j3 a0 | 0.500  | 0.500   | OK |
| traj_a1_s42    | a1      | 64000 | fd     | linear j4 a0 | 0.548  | 0.550   | OK |
| traj_a1_s42    | a1      | 64000 | tv     | linear a0    | 0.394  | 0.394   | OK |
| traj_a1_s42    | a1      | 64000 | tv     | linear a1    | 0.530  | 0.530   | OK |
| traj_a1_s42    | a1      | 64000 | tv     | clean a0     | 0.406  | 0.406   | OK |
| traj_a1_s42    | a1      | 64000 | tv     | clean a1     | 0.553  | 0.553   | OK |
| traj_a1_s42    | a1      | 64000 | tv     | linear j1 a0 | 0.378  | 0.378   | OK |
| traj_a1_s42    | a1      | 64000 | tv     | linear j2 a0 | 0.432  | 0.432   | OK |
| traj_a1_s42    | a1      | 64000 | tv     | linear j3 a0 | 0.490  | 0.490   | OK |
| traj_a1_s42    | swap65k | 8000  | fd     | linear a0    | 0.524  | 0.524   | OK |
| traj_a1_s42    | swap65k | 8000  | fd     | linear a1    | 0.522  | 0.522   | OK |
| traj_a1_s42    | swap65k | 8000  | fd     | clean a0     | 0.534  | 0.534   | OK |
| traj_a1_s42    | swap65k | 8000  | fd     | clean a1     | 0.531  | 0.531   | OK |
| traj_a1_s42    | swap65k | 8000  | fd     | linear j1 a0 | 0.496  | 0.496   | OK |
| traj_a1_s42    | swap65k | 8000  | fd     | linear j2 a0 | 0.519  | 0.519   | OK |
| traj_a1_s42    | swap65k | 8000  | fd     | linear j3 a0 | 0.517  | 0.517   | OK |
| traj_a1_s42    | swap65k | 8000  | fd     | linear j4 a0 | 0.508  | 0.508   | OK |
| traj_a1_s42    | swap65k | 8000  | tv     | linear a0    | 0.435  | 0.435   | OK |
| traj_a1_s42    | swap65k | 8000  | tv     | linear a1    | 0.498  | 0.498   | OK |
| traj_a1_s42    | swap65k | 8000  | tv     | clean a0     | 0.440  | 0.441   | OK |
| traj_a1_s42    | swap65k | 8000  | tv     | clean a1     | 0.519  | 0.519   | OK |
| traj_a1_s42    | swap65k | 8000  | tv     | linear j1 a0 | 0.458  | 0.458   | OK |
| traj_a1_s42    | swap65k | 8000  | tv     | linear j2 a0 | 0.475  | 0.475   | OK |
| traj_a1_s42    | swap65k | 8000  | tv     | linear j3 a0 | 0.493  | 0.493   | OK |
| traj_a1_s42    | swap65k | 8000  | tv     | linear j4 a0 | 0.530  | 0.530   | OK |
| traj_a1_s42    | swap65k | 24000 | fd     | linear a0    | 0.516  | 0.516   | OK |
| traj_a1_s42    | swap65k | 24000 | fd     | linear a1    | 0.516  | 0.516   | OK |
| traj_a1_s42    | swap65k | 24000 | fd     | clean a0     | 0.539  | 0.539   | OK |
| traj_a1_s42    | swap65k | 24000 | fd     | clean a1     | 0.536  | 0.536   | OK |
| traj_a1_s42    | swap65k | 24000 | fd     | linear j1 a0 | 0.500  | 0.500   | OK |
| traj_a1_s42    | swap65k | 24000 | fd     | linear j2 a0 | 0.516  | 0.516   | OK |
| traj_a1_s42    | swap65k | 24000 | fd     | linear j3 a0 | 0.505  | 0.505   | OK |
| traj_a1_s42    | swap65k | 24000 | fd     | linear j4 a0 | 0.500  | 0.500   | OK |
| traj_a1_s42    | swap65k | 24000 | tv     | linear a0    | 0.438  | 0.438   | OK |
| traj_a1_s42    | swap65k | 24000 | tv     | linear a1    | 0.496  | 0.496   | OK |
| traj_a1_s42    | swap65k | 24000 | tv     | clean a0     | 0.457  | 0.457   | OK |
| traj_a1_s42    | swap65k | 24000 | tv     | clean a1     | 0.535  | 0.534   | OK |
| traj_a1_s42    | swap65k | 24000 | tv     | linear j1 a0 | 0.480  | 0.480   | OK |
| traj_a1_s42    | swap65k | 24000 | tv     | linear j2 a0 | 0.465  | 0.465   | OK |
| traj_a1_s42    | swap65k | 24000 | tv     | linear j3 a0 | 0.487  | 0.487   | OK |
| traj_a1_s42    | swap65k | 24000 | tv     | linear j4 a0 | 0.500  | 0.500   | OK |
| traj_a1_s42    | swap65k | 64000 | fd     | linear a0    | 0.507  | 0.507   | OK |
| traj_a1_s42    | swap65k | 64000 | fd     | linear a1    | 0.499  | 0.499   | OK |
| traj_a1_s42    | swap65k | 64000 | fd     | clean a0     | 0.518  | 0.518   | OK |
| traj_a1_s42    | swap65k | 64000 | fd     | clean a1     | 0.534  | 0.534   | OK |
| traj_a1_s42    | swap65k | 64000 | fd     | linear j1 a0 | 0.501  | 0.501   | OK |
| traj_a1_s42    | swap65k | 64000 | fd     | linear j2 a0 | 0.515  | 0.515   | OK |
| traj_a1_s42    | swap65k | 64000 | fd     | linear j3 a0 | 0.512  | 0.512   | OK |
| traj_a1_s42    | swap65k | 64000 | fd     | linear j4 a0 | 0.496  | 0.496   | OK |
| traj_a1_s42    | swap65k | 64000 | tv     | linear a0    | 0.424  | 0.424   | OK |
| traj_a1_s42    | swap65k | 64000 | tv     | linear a1    | 0.463  | 0.463   | OK |
| traj_a1_s42    | swap65k | 64000 | tv     | clean a0     | 0.426  | 0.426   | OK |
| traj_a1_s42    | swap65k | 64000 | tv     | clean a1     | 0.543  | 0.543   | OK |
| traj_a1_s42    | swap65k | 64000 | tv     | linear j1 a0 | 0.468  | 0.468   | OK |
| traj_a1_s42    | swap65k | 64000 | tv     | linear j2 a0 | 0.473  | 0.473   | OK |
| traj_a1_s42    | swap65k | 64000 | tv     | linear j3 a0 | 0.456  | 0.456   | OK |
| traj_a1_s42    | swap65k | 64000 | tv     | linear j4 a0 | 0.439  | 0.439   | OK |
| traj_eps01_s42 | a1      | 64000 | fd     | linear a0    | 0.480  | 0.480   | OK |
| traj_eps01_s42 | a1      | 64000 | fd     | linear a1    | 0.494  | 0.494   | OK |
| traj_eps01_s42 | a1      | 64000 | fd     | clean a0     | 0.492  | 0.492   | OK |
| traj_eps01_s42 | a1      | 64000 | fd     | clean a1     | 0.499  | 0.499   | OK |
| traj_eps01_s42 | a1      | 64000 | fd     | linear j1 a0 | 0.501  | 0.501   | OK |
| traj_eps01_s42 | a1      | 64000 | fd     | linear j2 a0 | 0.480  | 0.480   | OK |
| traj_eps01_s42 | a1      | 64000 | fd     | linear j3 a0 | 0.502  | 0.502   | OK |
| traj_eps01_s42 | a1      | 64000 | fd     | linear j4 a0 | 0.494  | 0.494   | OK |
| traj_eps01_s42 | a1      | 64000 | tv     | linear a0    | 0.396  | 0.396   | OK |
| traj_eps01_s42 | a1      | 64000 | tv     | linear a1    | 0.457  | 0.457   | OK |
| traj_eps01_s42 | a1      | 64000 | tv     | clean a0     | 0.417  | 0.417   | OK |
| traj_eps01_s42 | a1      | 64000 | tv     | clean a1     | 0.474  | 0.474   | OK |
| traj_eps01_s42 | a1      | 64000 | tv     | linear j1 a0 | 0.535  | 0.537   | OK |
| traj_eps01_s42 | a1      | 64000 | tv     | linear j2 a0 | 0.523  | 0.523   | OK |
| traj_eps01_s42 | a1      | 64000 | tv     | linear j3 a0 | 0.542  | 0.542   | OK |
| traj_eps01_s42 | swap65k | 64000 | fd     | linear a0    | 0.477  | 0.477   | OK |
| traj_eps01_s42 | swap65k | 64000 | fd     | linear a1    | 0.505  | 0.505   | OK |
| traj_eps01_s42 | swap65k | 64000 | fd     | clean a0     | 0.501  | 0.501   | OK |
| traj_eps01_s42 | swap65k | 64000 | fd     | clean a1     | 0.528  | 0.528   | OK |
| traj_eps01_s42 | swap65k | 64000 | fd     | linear j1 a0 | 0.495  | 0.495   | OK |
| traj_eps01_s42 | swap65k | 64000 | fd     | linear j2 a0 | 0.518  | 0.518   | OK |
| traj_eps01_s42 | swap65k | 64000 | fd     | linear j3 a0 | 0.505  | 0.505   | OK |
| traj_eps01_s42 | swap65k | 64000 | fd     | linear j4 a0 | 0.504  | 0.504   | OK |
| traj_eps01_s42 | swap65k | 64000 | tv     | linear a0    | 0.384  | 0.384   | OK |
| traj_eps01_s42 | swap65k | 64000 | tv     | linear a1    | 0.465  | 0.465   | OK |
| traj_eps01_s42 | swap65k | 64000 | tv     | clean a0     | 0.414  | 0.414   | OK |
| traj_eps01_s42 | swap65k | 64000 | tv     | clean a1     | 0.472  | 0.472   | OK |
| traj_eps01_s42 | swap65k | 64000 | tv     | linear j1 a0 | 0.513  | 0.513   | OK |
| traj_eps01_s42 | swap65k | 64000 | tv     | linear j2 a0 | 0.483  | 0.483   | OK |
| traj_eps01_s42 | swap65k | 64000 | tv     | linear j3 a0 | 0.532  | 0.532   | OK |
| traj_eps01_s42 | swap65k | 64000 | tv     | linear j4 a0 | 0.606  | 0.606   | OK |

## Summary: the same contrast under three stratifications

`pooled` ranks within level over the whole held-out set; `by j` re-ranks inside each edit width; `by (j, k*)` re-ranks inside each exact (width, detection-depth) cell. `j` and `k*` both move the SCALE of the revision, so the pooled row ranks a wide or deep edit's large revisions against a narrow one's. `cov` is the share of the pooled rows the surviving `(j, k*)` cells retain (30-row floor); read only the rows where it is near 1.

| traj      | venue   | step  | anchor | critic | n pooled | pooled | by j  | by (j, k*) | cov  |
|-----------|---------|-------|--------|--------|----------|--------|-------|------------|------|
| a1_s42    | a1      | 8000  | onset  | linear | 1982     | 0.522  | 0.521 | 0.531      | 0.54 |
| a1_s42    | a1      | 8000  | onset  | mlp    | 1982     | 0.481  | 0.490 | 0.511      | 0.54 |
| a1_s42    | a1      | 8000  | onset  | clean  | 1982     | 0.525  | 0.523 | 0.530      | 0.54 |
| a1_s42    | a1      | 8000  | t_v    | linear | 465      | 0.482  | 0.493 | 0.490      | 0.93 |
| a1_s42    | a1      | 8000  | t_v    | mlp    | 465      | 0.482  | 0.502 | 0.516      | 0.93 |
| a1_s42    | a1      | 8000  | t_v    | clean  | 465      | 0.473  | 0.498 | 0.506      | 0.93 |
| a1_s42    | a1      | 24000 | onset  | linear | 1982     | 0.527  | 0.503 | 0.515      | 0.54 |
| a1_s42    | a1      | 24000 | onset  | mlp    | 1982     | 0.474  | 0.486 | 0.510      | 0.54 |
| a1_s42    | a1      | 24000 | onset  | clean  | 1982     | 0.536  | 0.512 | 0.512      | 0.54 |
| a1_s42    | a1      | 24000 | t_v    | linear | 465      | 0.424  | 0.461 | 0.462      | 0.93 |
| a1_s42    | a1      | 24000 | t_v    | mlp    | 465      | 0.437  | 0.476 | 0.477      | 0.93 |
| a1_s42    | a1      | 24000 | t_v    | clean  | 465      | 0.422  | 0.461 | 0.471      | 0.93 |
| a1_s42    | a1      | 64000 | onset  | linear | 1982     | 0.513  | 0.503 | 0.513      | 0.54 |
| a1_s42    | a1      | 64000 | onset  | mlp    | 1982     | 0.494  | 0.484 | 0.511      | 0.54 |
| a1_s42    | a1      | 64000 | onset  | clean  | 1982     | 0.514  | 0.500 | 0.522      | 0.54 |
| a1_s42    | a1      | 64000 | t_v    | linear | 465      | 0.394  | 0.422 | 0.446      | 0.93 |
| a1_s42    | a1      | 64000 | t_v    | mlp    | 465      | 0.434  | 0.440 | 0.452      | 0.93 |
| a1_s42    | a1      | 64000 | t_v    | clean  | 465      | 0.406  | 0.428 | 0.431      | 0.93 |
| a1_s42    | swap65k | 8000  | onset  | linear | 10308    | 0.524  | 0.511 | 0.508      | 0.84 |
| a1_s42    | swap65k | 8000  | onset  | mlp    | 10308    | 0.495  | 0.506 | 0.509      | 0.84 |
| a1_s42    | swap65k | 8000  | onset  | clean  | 10308    | 0.534  | 0.506 | 0.507      | 0.84 |
| a1_s42    | swap65k | 8000  | t_v    | linear | 3629     | 0.435  | 0.473 | 0.483      | 0.99 |
| a1_s42    | swap65k | 8000  | t_v    | mlp    | 3629     | 0.440  | 0.485 | 0.493      | 0.99 |
| a1_s42    | swap65k | 8000  | t_v    | clean  | 3629     | 0.441  | 0.468 | 0.481      | 0.99 |
| a1_s42    | swap65k | 24000 | onset  | linear | 10308    | 0.516  | 0.507 | 0.506      | 0.84 |
| a1_s42    | swap65k | 24000 | onset  | mlp    | 10308    | 0.491  | 0.503 | 0.504      | 0.84 |
| a1_s42    | swap65k | 24000 | onset  | clean  | 10308    | 0.539  | 0.507 | 0.507      | 0.84 |
| a1_s42    | swap65k | 24000 | t_v    | linear | 3629     | 0.438  | 0.476 | 0.499      | 0.99 |
| a1_s42    | swap65k | 24000 | t_v    | mlp    | 3629     | 0.440  | 0.491 | 0.497      | 0.99 |
| a1_s42    | swap65k | 24000 | t_v    | clean  | 3629     | 0.457  | 0.479 | 0.498      | 0.99 |
| a1_s42    | swap65k | 64000 | onset  | linear | 10308    | 0.507  | 0.509 | 0.509      | 0.84 |
| a1_s42    | swap65k | 64000 | onset  | mlp    | 10308    | 0.496  | 0.506 | 0.503      | 0.84 |
| a1_s42    | swap65k | 64000 | onset  | clean  | 10308    | 0.518  | 0.510 | 0.508      | 0.84 |
| a1_s42    | swap65k | 64000 | t_v    | linear | 3629     | 0.424  | 0.467 | 0.482      | 0.99 |
| a1_s42    | swap65k | 64000 | t_v    | mlp    | 3629     | 0.462  | 0.497 | 0.502      | 0.99 |
| a1_s42    | swap65k | 64000 | t_v    | clean  | 3629     | 0.426  | 0.454 | 0.469      | 0.99 |
| eps01_s42 | a1      | 64000 | onset  | linear | 1982     | 0.480  | 0.493 | 0.504      | 0.54 |
| eps01_s42 | a1      | 64000 | onset  | mlp    | 1982     | 0.464  | 0.471 | 0.500      | 0.54 |
| eps01_s42 | a1      | 64000 | onset  | clean  | 1982     | 0.492  | 0.490 | 0.500      | 0.54 |
| eps01_s42 | a1      | 64000 | t_v    | linear | 465      | 0.396  | 0.533 | 0.536      | 0.93 |
| eps01_s42 | a1      | 64000 | t_v    | mlp    | 465      | 0.434  | 0.493 | 0.499      | 0.93 |
| eps01_s42 | a1      | 64000 | t_v    | clean  | 465      | 0.417  | 0.509 | 0.513      | 0.93 |
| eps01_s42 | swap65k | 64000 | onset  | linear | 10308    | 0.477  | 0.506 | 0.505      | 0.84 |
| eps01_s42 | swap65k | 64000 | onset  | mlp    | 10308    | 0.470  | 0.506 | 0.505      | 0.84 |
| eps01_s42 | swap65k | 64000 | onset  | clean  | 10308    | 0.501  | 0.508 | 0.506      | 0.84 |
| eps01_s42 | swap65k | 64000 | t_v    | linear | 3629     | 0.384  | 0.508 | 0.518      | 0.99 |
| eps01_s42 | swap65k | 64000 | t_v    | mlp    | 3629     | 0.403  | 0.497 | 0.491      | 0.99 |
| eps01_s42 | swap65k | 64000 | t_v    | clean  | 3629     | 0.414  | 0.509 | 0.517      | 0.99 |

## `traj_a1_s42` / venue `a1` / step 8000 / anchor: edit onset (first_diff)

Critic block `post_block7`.

| critic | a | n    | rank cons | rank incons | win rate | scalar rank |
|--------|---|------|-----------|-------------|----------|-------------|
| linear | 0 | 1982 | 0.502     | 0.491       | 0.522    | 0.499       |
| linear | 1 | 1971 | 0.499     | 0.495       | 0.500    | 0.498       |
| mlp    | 0 | 1982 | 0.502     | 0.518       | 0.481    | 0.515       |
| mlp    | 1 | 1971 | 0.497     | 0.520       | 0.474    | 0.512       |
| clean  | 0 | 1982 | 0.505     | 0.489       | 0.525    | 0.500       |
| clean  | 1 | 1971 | 0.502     | 0.494       | 0.499    | 0.499       |

Per-`j` breakdown (a = 0):

| critic | j                 | n    | rank cons | rank incons | win rate |
|--------|-------------------|------|-----------|-------------|----------|
| linear | j1                | 371  | 0.532     | 0.514       | 0.582    |
| linear | j2                | 638  | 0.505     | 0.508       | 0.500    |
| linear | j3                | 662  | 0.494     | 0.496       | 0.495    |
| linear | j4                | 311  | 0.501     | 0.482       | 0.543    |
| linear | **all j (n-wtd)** | 1982 | --        | --          | 0.521    |
| mlp    | j1                | 371  | 0.541     | 0.554       | 0.496    |
| mlp    | j2                | 638  | 0.510     | 0.527       | 0.472    |
| mlp    | j3                | 662  | 0.491     | 0.508       | 0.494    |
| mlp    | j4                | 311  | 0.511     | 0.500       | 0.513    |
| mlp    | **all j (n-wtd)** | 1982 | --        | --          | 0.490    |
| clean  | j1                | 371  | 0.527     | 0.512       | 0.561    |
| clean  | j2                | 638  | 0.505     | 0.507       | 0.505    |
| clean  | j3                | 662  | 0.495     | 0.495       | 0.503    |
| clean  | j4                | 311  | 0.504     | 0.481       | 0.556    |
| clean  | **all j (n-wtd)** | 1982 | --        | --          | 0.523    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 195  | 0.500         | 0.500       | 0.554    |
| linear | j1k3                    | 97   | 0.500         | 0.500       | 0.557    |
| linear | j1k4                    | 46   | 0.500         | 0.500       | 0.587    |
| linear | j2k3                    | 229  | 0.501         | 0.499       | 0.504    |
| linear | j2k4                    | 110  | 0.502         | 0.500       | 0.505    |
| linear | j2k5                    | 30   | 0.497         | 0.495       | 0.500    |
| linear | j3k4                    | 209  | 0.502         | 0.498       | 0.529    |
| linear | j3k5                    | 123  | 0.502         | 0.492       | 0.520    |
| linear | j4k5                    | 22   | 0.458         | 0.430       | 0.636    |
| linear | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.531    |
| mlp    | j1k2                    | 195  | 0.500         | 0.500       | 0.487    |
| mlp    | j1k3                    | 97   | 0.500         | 0.500       | 0.505    |
| mlp    | j1k4                    | 46   | 0.500         | 0.500       | 0.522    |
| mlp    | j2k3                    | 229  | 0.499         | 0.500       | 0.502    |
| mlp    | j2k4                    | 110  | 0.501         | 0.500       | 0.536    |
| mlp    | j2k5                    | 30   | 0.498         | 0.499       | 0.517    |
| mlp    | j3k4                    | 209  | 0.502         | 0.499       | 0.526    |
| mlp    | j3k5                    | 123  | 0.500         | 0.498       | 0.512    |
| mlp    | j4k5                    | 22   | 0.487         | 0.460       | 0.545    |
| mlp    | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.511    |
| clean  | j1k2                    | 195  | 0.500         | 0.500       | 0.528    |
| clean  | j1k3                    | 97   | 0.500         | 0.500       | 0.526    |
| clean  | j1k4                    | 46   | 0.500         | 0.500       | 0.565    |
| clean  | j2k3                    | 229  | 0.501         | 0.499       | 0.498    |
| clean  | j2k4                    | 110  | 0.502         | 0.500       | 0.527    |
| clean  | j2k5                    | 30   | 0.497         | 0.494       | 0.433    |
| clean  | j3k4                    | 209  | 0.502         | 0.496       | 0.555    |
| clean  | j3k5                    | 123  | 0.502         | 0.494       | 0.545    |
| clean  | j4k5                    | 22   | 0.464         | 0.421       | 0.636    |
| clean  | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.530    |

## `traj_a1_s42` / venue `a1` / step 8000 / anchor: t_v (Bayesian-detectable violation)

Critic block `post_block7`.

| critic | a | n   | rank cons | rank incons | win rate | scalar rank |
|--------|---|-----|-----------|-------------|----------|-------------|
| linear | 0 | 465 | 0.495     | 0.520       | 0.482    | 0.511       |
| linear | 1 | 304 | 0.512     | 0.501       | 0.503    | 0.506       |
| mlp    | 0 | 465 | 0.504     | 0.538       | 0.482    | 0.524       |
| mlp    | 1 | 304 | 0.506     | 0.528       | 0.480    | 0.521       |
| clean  | 0 | 465 | 0.496     | 0.521       | 0.473    | 0.513       |
| clean  | 1 | 304 | 0.520     | 0.511       | 0.500    | 0.513       |

Per-`j` breakdown (a = 0):

| critic | j                 | n   | rank cons | rank incons | win rate |
|--------|-------------------|-----|-----------|-------------|----------|
| linear | j1                | 188 | 0.514     | 0.517       | 0.479    |
| linear | j2                | 176 | 0.511     | 0.541       | 0.466    |
| linear | j3                | 96  | 0.518     | 0.497       | 0.573    |
| linear | **all j (n-wtd)** | 460 | --        | --          | 0.493    |
| mlp    | j1                | 188 | 0.525     | 0.531       | 0.511    |
| mlp    | j2                | 176 | 0.518     | 0.547       | 0.438    |
| mlp    | j3                | 96  | 0.498     | 0.461       | 0.604    |
| mlp    | **all j (n-wtd)** | 460 | --        | --          | 0.502    |
| clean  | j1                | 188 | 0.510     | 0.517       | 0.484    |
| clean  | j2                | 176 | 0.511     | 0.537       | 0.472    |
| clean  | j3                | 96  | 0.516     | 0.499       | 0.573    |
| clean  | **all j (n-wtd)** | 460 | --        | --          | 0.498    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n   | rank cons     | rank incons | win rate |
|--------|-------------------------|-----|---------------|-------------|----------|
| linear | j1k2                    | 108 | 0.509         | 0.519       | 0.444    |
| linear | j1k3                    | 59  | 0.529         | 0.510       | 0.525    |
| linear | j2k3                    | 129 | 0.490         | 0.520       | 0.473    |
| linear | j2k4                    | 41  | 0.536         | 0.534       | 0.415    |
| linear | j3k4                    | 74  | 0.531         | 0.511       | 0.595    |
| linear | j3k5                    | 22  | 0.491         | 0.469       | 0.500    |
| linear | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.490    |
| mlp    | j1k2                    | 108 | 0.514         | 0.524       | 0.491    |
| mlp    | j1k3                    | 59  | 0.554         | 0.537       | 0.576    |
| mlp    | j2k3                    | 129 | 0.504         | 0.522       | 0.450    |
| mlp    | j2k4                    | 41  | 0.507         | 0.533       | 0.524    |
| mlp    | j3k4                    | 74  | 0.517         | 0.459       | 0.595    |
| mlp    | j3k5                    | 22  | 0.433         | 0.452       | 0.591    |
| mlp    | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.516    |
| clean  | j1k2                    | 108 | 0.510         | 0.523       | 0.463    |
| clean  | j1k3                    | 59  | 0.517         | 0.508       | 0.559    |
| clean  | j2k3                    | 129 | 0.490         | 0.516       | 0.484    |
| clean  | j2k4                    | 41  | 0.536         | 0.530       | 0.451    |
| clean  | j3k4                    | 74  | 0.530         | 0.513       | 0.568    |
| clean  | j3k5                    | 22  | 0.486         | 0.472       | 0.591    |
| clean  | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.506    |

## `traj_a1_s42` / venue `a1` / step 24000 / anchor: edit onset (first_diff)

Critic block `post_block7`.

| critic | a | n    | rank cons | rank incons | win rate | scalar rank |
|--------|---|------|-----------|-------------|----------|-------------|
| linear | 0 | 1982 | 0.499     | 0.490       | 0.527    | 0.496       |
| linear | 1 | 1971 | 0.496     | 0.496       | 0.509    | 0.497       |
| mlp    | 0 | 1982 | 0.498     | 0.521       | 0.474    | 0.515       |
| mlp    | 1 | 1971 | 0.501     | 0.521       | 0.454    | 0.514       |
| clean  | 0 | 1982 | 0.503     | 0.486       | 0.536    | 0.497       |
| clean  | 1 | 1971 | 0.499     | 0.496       | 0.516    | 0.500       |

Per-`j` breakdown (a = 0):

| critic | j                 | n    | rank cons | rank incons | win rate |
|--------|-------------------|------|-----------|-------------|----------|
| linear | j1                | 371  | 0.519     | 0.505       | 0.542    |
| linear | j2                | 638  | 0.504     | 0.511       | 0.469    |
| linear | j3                | 662  | 0.491     | 0.494       | 0.491    |
| linear | j4                | 311  | 0.504     | 0.481       | 0.553    |
| linear | **all j (n-wtd)** | 1982 | --        | --          | 0.503    |
| mlp    | j1                | 371  | 0.540     | 0.571       | 0.485    |
| mlp    | j2                | 638  | 0.511     | 0.531       | 0.486    |
| mlp    | j3                | 662  | 0.489     | 0.502       | 0.486    |
| mlp    | j4                | 311  | 0.505     | 0.508       | 0.489    |
| mlp    | **all j (n-wtd)** | 1982 | --        | --          | 0.486    |
| clean  | j1                | 371  | 0.511     | 0.495       | 0.550    |
| clean  | j2                | 638  | 0.503     | 0.508       | 0.495    |
| clean  | j3                | 662  | 0.491     | 0.493       | 0.494    |
| clean  | j4                | 311  | 0.509     | 0.485       | 0.537    |
| clean  | **all j (n-wtd)** | 1982 | --        | --          | 0.512    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 195  | 0.500         | 0.500       | 0.528    |
| linear | j1k3                    | 97   | 0.500         | 0.500       | 0.515    |
| linear | j1k4                    | 46   | 0.500         | 0.500       | 0.522    |
| linear | j2k3                    | 229  | 0.502         | 0.499       | 0.474    |
| linear | j2k4                    | 110  | 0.503         | 0.500       | 0.514    |
| linear | j2k5                    | 30   | 0.509         | 0.496       | 0.583    |
| linear | j3k4                    | 209  | 0.501         | 0.497       | 0.531    |
| linear | j3k5                    | 123  | 0.502         | 0.497       | 0.496    |
| linear | j4k5                    | 22   | 0.563         | 0.483       | 0.682    |
| linear | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.515    |
| mlp    | j1k2                    | 195  | 0.500         | 0.500       | 0.518    |
| mlp    | j1k3                    | 97   | 0.500         | 0.500       | 0.505    |
| mlp    | j1k4                    | 46   | 0.500         | 0.500       | 0.565    |
| mlp    | j2k3                    | 229  | 0.499         | 0.501       | 0.500    |
| mlp    | j2k4                    | 110  | 0.502         | 0.499       | 0.527    |
| mlp    | j2k5                    | 30   | 0.500         | 0.498       | 0.467    |
| mlp    | j3k4                    | 209  | 0.502         | 0.496       | 0.522    |
| mlp    | j3k5                    | 123  | 0.499         | 0.502       | 0.488    |
| mlp    | j4k5                    | 22   | 0.503         | 0.518       | 0.455    |
| mlp    | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.510    |
| clean  | j1k2                    | 195  | 0.500         | 0.500       | 0.523    |
| clean  | j1k3                    | 97   | 0.500         | 0.500       | 0.459    |
| clean  | j1k4                    | 46   | 0.500         | 0.500       | 0.543    |
| clean  | j2k3                    | 229  | 0.503         | 0.499       | 0.507    |
| clean  | j2k4                    | 110  | 0.503         | 0.500       | 0.536    |
| clean  | j2k5                    | 30   | 0.510         | 0.495       | 0.517    |
| clean  | j3k4                    | 209  | 0.501         | 0.497       | 0.510    |
| clean  | j3k5                    | 123  | 0.502         | 0.499       | 0.504    |
| clean  | j4k5                    | 22   | 0.549         | 0.489       | 0.591    |
| clean  | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.512    |

## `traj_a1_s42` / venue `a1` / step 24000 / anchor: t_v (Bayesian-detectable violation)

Critic block `post_block7`.

| critic | a | n   | rank cons | rank incons | win rate | scalar rank |
|--------|---|-----|-----------|-------------|----------|-------------|
| linear | 0 | 465 | 0.485     | 0.536       | 0.424    | 0.511       |
| linear | 1 | 304 | 0.499     | 0.518       | 0.500    | 0.504       |
| mlp    | 0 | 465 | 0.510     | 0.556       | 0.437    | 0.535       |
| mlp    | 1 | 304 | 0.504     | 0.569       | 0.428    | 0.540       |
| clean  | 0 | 465 | 0.482     | 0.530       | 0.422    | 0.509       |
| clean  | 1 | 304 | 0.527     | 0.530       | 0.516    | 0.522       |

Per-`j` breakdown (a = 0):

| critic | j                 | n   | rank cons | rank incons | win rate |
|--------|-------------------|-----|-----------|-------------|----------|
| linear | j1                | 188 | 0.494     | 0.502       | 0.473    |
| linear | j2                | 176 | 0.502     | 0.535       | 0.477    |
| linear | j3                | 96  | 0.502     | 0.541       | 0.406    |
| linear | **all j (n-wtd)** | 460 | --        | --          | 0.461    |
| mlp    | j1                | 188 | 0.515     | 0.533       | 0.484    |
| mlp    | j2                | 176 | 0.514     | 0.538       | 0.460    |
| mlp    | j3                | 96  | 0.497     | 0.484       | 0.490    |
| mlp    | **all j (n-wtd)** | 460 | --        | --          | 0.476    |
| clean  | j1                | 188 | 0.481     | 0.492       | 0.471    |
| clean  | j2                | 176 | 0.500     | 0.533       | 0.474    |
| clean  | j3                | 96  | 0.494     | 0.549       | 0.417    |
| clean  | **all j (n-wtd)** | 460 | --        | --          | 0.461    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n   | rank cons     | rank incons | win rate |
|--------|-------------------------|-----|---------------|-------------|----------|
| linear | j1k2                    | 108 | 0.491         | 0.488       | 0.444    |
| linear | j1k3                    | 59  | 0.509         | 0.517       | 0.475    |
| linear | j2k3                    | 129 | 0.495         | 0.520       | 0.473    |
| linear | j2k4                    | 41  | 0.483         | 0.498       | 0.512    |
| linear | j3k4                    | 74  | 0.515         | 0.534       | 0.446    |
| linear | j3k5                    | 22  | 0.450         | 0.535       | 0.409    |
| linear | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.462    |
| mlp    | j1k2                    | 108 | 0.497         | 0.513       | 0.481    |
| mlp    | j1k3                    | 59  | 0.537         | 0.546       | 0.475    |
| mlp    | j2k3                    | 129 | 0.496         | 0.508       | 0.508    |
| mlp    | j2k4                    | 41  | 0.504         | 0.544       | 0.341    |
| mlp    | j3k4                    | 74  | 0.502         | 0.464       | 0.527    |
| mlp    | j3k5                    | 22  | 0.439         | 0.492       | 0.364    |
| mlp    | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.477    |
| clean  | j1k2                    | 108 | 0.490         | 0.484       | 0.454    |
| clean  | j1k3                    | 59  | 0.488         | 0.500       | 0.475    |
| clean  | j2k3                    | 129 | 0.494         | 0.517       | 0.488    |
| clean  | j2k4                    | 41  | 0.496         | 0.509       | 0.561    |
| clean  | j3k4                    | 74  | 0.503         | 0.535       | 0.459    |
| clean  | j3k5                    | 22  | 0.461         | 0.557       | 0.318    |
| clean  | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.471    |

## `traj_a1_s42` / venue `a1` / step 64000 / anchor: edit onset (first_diff)

Critic block `post_block7`.

| critic | a | n    | rank cons | rank incons | win rate | scalar rank |
|--------|---|------|-----------|-------------|----------|-------------|
| linear | 0 | 1982 | 0.495     | 0.498       | 0.513    | 0.499       |
| linear | 1 | 1971 | 0.497     | 0.503       | 0.506    | 0.498       |
| mlp    | 0 | 1982 | 0.503     | 0.519       | 0.494    | 0.516       |
| mlp    | 1 | 1971 | 0.504     | 0.523       | 0.472    | 0.514       |
| clean  | 0 | 1982 | 0.496     | 0.495       | 0.514    | 0.498       |
| clean  | 1 | 1971 | 0.502     | 0.500       | 0.516    | 0.499       |

Per-`j` breakdown (a = 0):

| critic | j                 | n    | rank cons | rank incons | win rate |
|--------|-------------------|------|-----------|-------------|----------|
| linear | j1                | 371  | 0.513     | 0.520       | 0.491    |
| linear | j2                | 638  | 0.500     | 0.505       | 0.489    |
| linear | j3                | 662  | 0.492     | 0.502       | 0.500    |
| linear | j4                | 311  | 0.505     | 0.502       | 0.550    |
| linear | **all j (n-wtd)** | 1982 | --        | --          | 0.503    |
| mlp    | j1                | 371  | 0.545     | 0.580       | 0.453    |
| mlp    | j2                | 638  | 0.514     | 0.524       | 0.478    |
| mlp    | j3                | 662  | 0.492     | 0.510       | 0.494    |
| mlp    | j4                | 311  | 0.520     | 0.535       | 0.514    |
| mlp    | **all j (n-wtd)** | 1982 | --        | --          | 0.484    |
| clean  | j1                | 371  | 0.499     | 0.509       | 0.492    |
| clean  | j2                | 638  | 0.500     | 0.503       | 0.503    |
| clean  | j3                | 662  | 0.489     | 0.501       | 0.491    |
| clean  | j4                | 311  | 0.502     | 0.510       | 0.524    |
| clean  | **all j (n-wtd)** | 1982 | --        | --          | 0.500    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 195  | 0.500         | 0.500       | 0.505    |
| linear | j1k3                    | 97   | 0.500         | 0.500       | 0.526    |
| linear | j1k4                    | 46   | 0.500         | 0.500       | 0.543    |
| linear | j2k3                    | 229  | 0.502         | 0.499       | 0.472    |
| linear | j2k4                    | 110  | 0.501         | 0.500       | 0.532    |
| linear | j2k5                    | 30   | 0.508         | 0.499       | 0.500    |
| linear | j3k4                    | 209  | 0.500         | 0.501       | 0.543    |
| linear | j3k5                    | 123  | 0.501         | 0.496       | 0.496    |
| linear | j4k5                    | 22   | 0.527         | 0.482       | 0.636    |
| linear | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.513    |
| mlp    | j1k2                    | 195  | 0.500         | 0.500       | 0.482    |
| mlp    | j1k3                    | 97   | 0.500         | 0.500       | 0.505    |
| mlp    | j1k4                    | 46   | 0.500         | 0.500       | 0.478    |
| mlp    | j2k3                    | 229  | 0.500         | 0.501       | 0.520    |
| mlp    | j2k4                    | 110  | 0.501         | 0.500       | 0.514    |
| mlp    | j2k5                    | 30   | 0.501         | 0.496       | 0.483    |
| mlp    | j3k4                    | 209  | 0.502         | 0.497       | 0.531    |
| mlp    | j3k5                    | 123  | 0.501         | 0.497       | 0.545    |
| mlp    | j4k5                    | 22   | 0.482         | 0.505       | 0.409    |
| mlp    | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.511    |
| clean  | j1k2                    | 195  | 0.500         | 0.500       | 0.497    |
| clean  | j1k3                    | 97   | 0.500         | 0.500       | 0.557    |
| clean  | j1k4                    | 46   | 0.500         | 0.500       | 0.522    |
| clean  | j2k3                    | 229  | 0.503         | 0.498       | 0.493    |
| clean  | j2k4                    | 110  | 0.503         | 0.500       | 0.545    |
| clean  | j2k5                    | 30   | 0.507         | 0.497       | 0.583    |
| clean  | j3k4                    | 209  | 0.500         | 0.502       | 0.531    |
| clean  | j3k5                    | 123  | 0.500         | 0.498       | 0.488    |
| clean  | j4k5                    | 22   | 0.525         | 0.489       | 0.773    |
| clean  | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.522    |

## `traj_a1_s42` / venue `a1` / step 64000 / anchor: t_v (Bayesian-detectable violation)

Critic block `post_block7`.

| critic | a | n   | rank cons | rank incons | win rate | scalar rank |
|--------|---|-----|-----------|-------------|----------|-------------|
| linear | 0 | 465 | 0.486     | 0.533       | 0.394    | 0.515       |
| linear | 1 | 304 | 0.498     | 0.491       | 0.530    | 0.485       |
| mlp    | 0 | 465 | 0.527     | 0.561       | 0.434    | 0.547       |
| mlp    | 1 | 304 | 0.500     | 0.549       | 0.431    | 0.523       |
| clean  | 0 | 465 | 0.478     | 0.528       | 0.406    | 0.510       |
| clean  | 1 | 304 | 0.537     | 0.518       | 0.553    | 0.515       |

Per-`j` breakdown (a = 0):

| critic | j                 | n   | rank cons | rank incons | win rate |
|--------|-------------------|-----|-----------|-------------|----------|
| linear | j1                | 188 | 0.479     | 0.516       | 0.378    |
| linear | j2                | 176 | 0.513     | 0.542       | 0.432    |
| linear | j3                | 96  | 0.514     | 0.522       | 0.490    |
| linear | **all j (n-wtd)** | 460 | --        | --          | 0.422    |
| mlp    | j1                | 188 | 0.514     | 0.530       | 0.444    |
| mlp    | j2                | 176 | 0.523     | 0.557       | 0.386    |
| mlp    | j3                | 96  | 0.503     | 0.484       | 0.531    |
| mlp    | **all j (n-wtd)** | 460 | --        | --          | 0.440    |
| clean  | j1                | 188 | 0.464     | 0.511       | 0.388    |
| clean  | j2                | 176 | 0.511     | 0.538       | 0.455    |
| clean  | j3                | 96  | 0.504     | 0.530       | 0.458    |
| clean  | **all j (n-wtd)** | 460 | --        | --          | 0.428    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n   | rank cons     | rank incons | win rate |
|--------|-------------------------|-----|---------------|-------------|----------|
| linear | j1k2                    | 108 | 0.473         | 0.510       | 0.389    |
| linear | j1k3                    | 59  | 0.525         | 0.508       | 0.475    |
| linear | j2k3                    | 129 | 0.496         | 0.528       | 0.442    |
| linear | j2k4                    | 41  | 0.524         | 0.523       | 0.439    |
| linear | j3k4                    | 74  | 0.515         | 0.494       | 0.527    |
| linear | j3k5                    | 22  | 0.493         | 0.557       | 0.409    |
| linear | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.446    |
| mlp    | j1k2                    | 108 | 0.497         | 0.518       | 0.398    |
| mlp    | j1k3                    | 59  | 0.546         | 0.542       | 0.525    |
| mlp    | j2k3                    | 129 | 0.501         | 0.539       | 0.395    |
| mlp    | j2k4                    | 41  | 0.512         | 0.538       | 0.427    |
| mlp    | j3k4                    | 74  | 0.491         | 0.470       | 0.568    |
| mlp    | j3k5                    | 22  | 0.471         | 0.434       | 0.500    |
| mlp    | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.452    |
| clean  | j1k2                    | 108 | 0.457         | 0.512       | 0.343    |
| clean  | j1k3                    | 59  | 0.512         | 0.495       | 0.483    |
| clean  | j2k3                    | 129 | 0.495         | 0.525       | 0.434    |
| clean  | j2k4                    | 41  | 0.533         | 0.518       | 0.488    |
| clean  | j3k4                    | 74  | 0.506         | 0.502       | 0.500    |
| clean  | j3k5                    | 22  | 0.488         | 0.560       | 0.364    |
| clean  | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.431    |

## `traj_a1_s42` / venue `swap65k` / step 8000 / anchor: edit onset (first_diff)

Critic block `post_block7`.

| critic | a | n     | rank cons | rank incons | win rate | scalar rank |
|--------|---|-------|-----------|-------------|----------|-------------|
| linear | 0 | 10308 | 0.494     | 0.485       | 0.524    | 0.490       |
| linear | 1 | 10126 | 0.503     | 0.496       | 0.522    | 0.498       |
| mlp    | 0 | 10308 | 0.492     | 0.503       | 0.495    | 0.498       |
| mlp    | 1 | 10126 | 0.496     | 0.505       | 0.491    | 0.499       |
| clean  | 0 | 10308 | 0.501     | 0.483       | 0.534    | 0.493       |
| clean  | 1 | 10126 | 0.511     | 0.496       | 0.531    | 0.502       |

Per-`j` breakdown (a = 0):

| critic | j                 | n     | rank cons | rank incons | win rate |
|--------|-------------------|-------|-----------|-------------|----------|
| linear | j1                | 3140  | 0.500     | 0.500       | 0.496    |
| linear | j2                | 3356  | 0.500     | 0.500       | 0.519    |
| linear | j3                | 3320  | 0.500     | 0.498       | 0.517    |
| linear | j4                | 492   | 0.494     | 0.487       | 0.508    |
| linear | **all j (n-wtd)** | 10308 | --        | --          | 0.511    |
| mlp    | j1                | 3140  | 0.500     | 0.500       | 0.511    |
| mlp    | j2                | 3356  | 0.500     | 0.500       | 0.507    |
| mlp    | j3                | 3320  | 0.500     | 0.499       | 0.502    |
| mlp    | j4                | 492   | 0.491     | 0.481       | 0.502    |
| mlp    | **all j (n-wtd)** | 10308 | --        | --          | 0.506    |
| clean  | j1                | 3140  | 0.500     | 0.500       | 0.500    |
| clean  | j2                | 3356  | 0.500     | 0.500       | 0.510    |
| clean  | j3                | 3320  | 0.500     | 0.499       | 0.507    |
| clean  | j4                | 492   | 0.495     | 0.492       | 0.510    |
| clean  | **all j (n-wtd)** | 10308 | --        | --          | 0.506    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 1542 | 0.500         | 0.500       | 0.490    |
| linear | j1k3                    | 962  | 0.500         | 0.500       | 0.502    |
| linear | j1k4                    | 366  | 0.500         | 0.500       | 0.495    |
| linear | j1k5                    | 90   | 0.500         | 0.500       | 0.461    |
| linear | j2k3                    | 1751 | 0.500         | 0.500       | 0.521    |
| linear | j2k4                    | 911  | 0.501         | 0.500       | 0.502    |
| linear | j2k5                    | 328  | 0.499         | 0.500       | 0.524    |
| linear | j3k4                    | 1609 | 0.501         | 0.499       | 0.528    |
| linear | j3k5                    | 911  | 0.499         | 0.499       | 0.503    |
| linear | j4k5                    | 236  | 0.521         | 0.518       | 0.466    |
| linear | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.508    |
| mlp    | j1k2                    | 1542 | 0.500         | 0.500       | 0.520    |
| mlp    | j1k3                    | 962  | 0.500         | 0.500       | 0.499    |
| mlp    | j1k4                    | 366  | 0.500         | 0.500       | 0.536    |
| mlp    | j1k5                    | 90   | 0.500         | 0.500       | 0.511    |
| mlp    | j2k3                    | 1751 | 0.500         | 0.500       | 0.502    |
| mlp    | j2k4                    | 911  | 0.501         | 0.500       | 0.523    |
| mlp    | j2k5                    | 328  | 0.501         | 0.499       | 0.495    |
| mlp    | j3k4                    | 1609 | 0.500         | 0.499       | 0.508    |
| mlp    | j3k5                    | 911  | 0.499         | 0.500       | 0.498    |
| mlp    | j4k5                    | 236  | 0.513         | 0.518       | 0.500    |
| mlp    | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.509    |
| clean  | j1k2                    | 1542 | 0.500         | 0.500       | 0.505    |
| clean  | j1k3                    | 962  | 0.500         | 0.500       | 0.492    |
| clean  | j1k4                    | 366  | 0.500         | 0.500       | 0.516    |
| clean  | j1k5                    | 90   | 0.500         | 0.500       | 0.500    |
| clean  | j2k3                    | 1751 | 0.500         | 0.501       | 0.518    |
| clean  | j2k4                    | 911  | 0.501         | 0.500       | 0.498    |
| clean  | j2k5                    | 328  | 0.499         | 0.500       | 0.530    |
| clean  | j3k4                    | 1609 | 0.501         | 0.500       | 0.515    |
| clean  | j3k5                    | 911  | 0.500         | 0.498       | 0.498    |
| clean  | j4k5                    | 236  | 0.522         | 0.521       | 0.483    |
| clean  | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.507    |

## `traj_a1_s42` / venue `swap65k` / step 8000 / anchor: t_v (Bayesian-detectable violation)

Critic block `post_block7`.

| critic | a | n    | rank cons | rank incons | win rate | scalar rank |
|--------|---|------|-----------|-------------|----------|-------------|
| linear | 0 | 3629 | 0.485     | 0.523       | 0.435    | 0.508       |
| linear | 1 | 2452 | 0.507     | 0.512       | 0.498    | 0.510       |
| mlp    | 0 | 3629 | 0.500     | 0.550       | 0.440    | 0.528       |
| mlp    | 1 | 2452 | 0.502     | 0.542       | 0.462    | 0.526       |
| clean  | 0 | 3629 | 0.485     | 0.523       | 0.441    | 0.510       |
| clean  | 1 | 2452 | 0.520     | 0.512       | 0.519    | 0.515       |

Per-`j` breakdown (a = 0):

| critic | j                 | n    | rank cons | rank incons | win rate |
|--------|-------------------|------|-----------|-------------|----------|
| linear | j1                | 1512 | 0.519     | 0.533       | 0.458    |
| linear | j2                | 1295 | 0.512     | 0.529       | 0.475    |
| linear | j3                | 756  | 0.508     | 0.516       | 0.493    |
| linear | j4                | 66   | 0.494     | 0.479       | 0.530    |
| linear | **all j (n-wtd)** | 3629 | --        | --          | 0.473    |
| mlp    | j1                | 1512 | 0.537     | 0.548       | 0.492    |
| mlp    | j2                | 1295 | 0.518     | 0.538       | 0.463    |
| mlp    | j3                | 756  | 0.498     | 0.496       | 0.505    |
| mlp    | j4                | 66   | 0.453     | 0.438       | 0.530    |
| mlp    | **all j (n-wtd)** | 3629 | --        | --          | 0.485    |
| clean  | j1                | 1512 | 0.508     | 0.527       | 0.455    |
| clean  | j2                | 1295 | 0.508     | 0.531       | 0.474    |
| clean  | j3                | 756  | 0.510     | 0.521       | 0.479    |
| clean  | j4                | 66   | 0.508     | 0.491       | 0.500    |
| clean  | **all j (n-wtd)** | 3629 | --        | --          | 0.468    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 864  | 0.505         | 0.526       | 0.454    |
| linear | j1k3                    | 482  | 0.522         | 0.516       | 0.479    |
| linear | j1k4                    | 147  | 0.545         | 0.534       | 0.524    |
| linear | j2k3                    | 946  | 0.501         | 0.505       | 0.498    |
| linear | j2k4                    | 300  | 0.530         | 0.552       | 0.450    |
| linear | j2k5                    | 49   | 0.554         | 0.495       | 0.633    |
| linear | j3k4                    | 619  | 0.512         | 0.520       | 0.499    |
| linear | j3k5                    | 137  | 0.512         | 0.510       | 0.467    |
| linear | j4k5                    | 66   | 0.495         | 0.479       | 0.530    |
| linear | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.483    |
| mlp    | j1k2                    | 864  | 0.510         | 0.528       | 0.471    |
| mlp    | j1k3                    | 482  | 0.551         | 0.549       | 0.519    |
| mlp    | j1k4                    | 147  | 0.566         | 0.556       | 0.565    |
| mlp    | j2k3                    | 946  | 0.499         | 0.504       | 0.484    |
| mlp    | j2k4                    | 300  | 0.536         | 0.564       | 0.463    |
| mlp    | j2k5                    | 49   | 0.535         | 0.525       | 0.469    |
| mlp    | j3k4                    | 619  | 0.504         | 0.501       | 0.517    |
| mlp    | j3k5                    | 137  | 0.486         | 0.473       | 0.489    |
| mlp    | j4k5                    | 66   | 0.453         | 0.437       | 0.530    |
| mlp    | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.493    |
| clean  | j1k2                    | 864  | 0.502         | 0.525       | 0.455    |
| clean  | j1k3                    | 482  | 0.507         | 0.504       | 0.485    |
| clean  | j1k4                    | 147  | 0.526         | 0.532       | 0.524    |
| clean  | j2k3                    | 946  | 0.501         | 0.505       | 0.504    |
| clean  | j2k4                    | 300  | 0.515         | 0.548       | 0.437    |
| clean  | j2k5                    | 49   | 0.536         | 0.497       | 0.633    |
| clean  | j3k4                    | 619  | 0.512         | 0.519       | 0.491    |
| clean  | j3k5                    | 137  | 0.507         | 0.524       | 0.423    |
| clean  | j4k5                    | 66   | 0.508         | 0.490       | 0.500    |
| clean  | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.481    |

## `traj_a1_s42` / venue `swap65k` / step 24000 / anchor: edit onset (first_diff)

Critic block `post_block7`.

| critic | a | n     | rank cons | rank incons | win rate | scalar rank |
|--------|---|-------|-----------|-------------|----------|-------------|
| linear | 0 | 10308 | 0.491     | 0.485       | 0.516    | 0.488       |
| linear | 1 | 10126 | 0.504     | 0.499       | 0.516    | 0.500       |
| mlp    | 0 | 10308 | 0.490     | 0.501       | 0.491    | 0.496       |
| mlp    | 1 | 10126 | 0.496     | 0.505       | 0.490    | 0.499       |
| clean  | 0 | 10308 | 0.501     | 0.481       | 0.539    | 0.493       |
| clean  | 1 | 10126 | 0.519     | 0.500       | 0.536    | 0.508       |

Per-`j` breakdown (a = 0):

| critic | j                 | n     | rank cons | rank incons | win rate |
|--------|-------------------|-------|-----------|-------------|----------|
| linear | j1                | 3140  | 0.500     | 0.500       | 0.500    |
| linear | j2                | 3356  | 0.501     | 0.500       | 0.516    |
| linear | j3                | 3320  | 0.499     | 0.500       | 0.505    |
| linear | j4                | 492   | 0.483     | 0.468       | 0.500    |
| linear | **all j (n-wtd)** | 10308 | --        | --          | 0.507    |
| mlp    | j1                | 3140  | 0.500     | 0.500       | 0.489    |
| mlp    | j2                | 3356  | 0.500     | 0.500       | 0.513    |
| mlp    | j3                | 3320  | 0.500     | 0.500       | 0.503    |
| mlp    | j4                | 492   | 0.485     | 0.468       | 0.524    |
| mlp    | **all j (n-wtd)** | 10308 | --        | --          | 0.503    |
| clean  | j1                | 3140  | 0.500     | 0.500       | 0.511    |
| clean  | j2                | 3356  | 0.501     | 0.500       | 0.510    |
| clean  | j3                | 3320  | 0.499     | 0.501       | 0.502    |
| clean  | j4                | 492   | 0.490     | 0.479       | 0.492    |
| clean  | **all j (n-wtd)** | 10308 | --        | --          | 0.507    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 1542 | 0.500         | 0.500       | 0.494    |
| linear | j1k3                    | 962  | 0.500         | 0.500       | 0.500    |
| linear | j1k4                    | 366  | 0.500         | 0.500       | 0.533    |
| linear | j1k5                    | 90   | 0.500         | 0.500       | 0.467    |
| linear | j2k3                    | 1751 | 0.500         | 0.501       | 0.518    |
| linear | j2k4                    | 911  | 0.502         | 0.499       | 0.503    |
| linear | j2k5                    | 328  | 0.500         | 0.500       | 0.549    |
| linear | j3k4                    | 1609 | 0.498         | 0.502       | 0.504    |
| linear | j3k5                    | 911  | 0.500         | 0.499       | 0.507    |
| linear | j4k5                    | 236  | 0.486         | 0.491       | 0.441    |
| linear | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.506    |
| mlp    | j1k2                    | 1542 | 0.500         | 0.500       | 0.493    |
| mlp    | j1k3                    | 962  | 0.500         | 0.500       | 0.493    |
| mlp    | j1k4                    | 366  | 0.500         | 0.500       | 0.516    |
| mlp    | j1k5                    | 90   | 0.500         | 0.500       | 0.500    |
| mlp    | j2k3                    | 1751 | 0.500         | 0.500       | 0.511    |
| mlp    | j2k4                    | 911  | 0.501         | 0.500       | 0.508    |
| mlp    | j2k5                    | 328  | 0.501         | 0.500       | 0.509    |
| mlp    | j3k4                    | 1609 | 0.500         | 0.501       | 0.507    |
| mlp    | j3k5                    | 911  | 0.500         | 0.500       | 0.503    |
| mlp    | j4k5                    | 236  | 0.490         | 0.485       | 0.517    |
| mlp    | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.504    |
| clean  | j1k2                    | 1542 | 0.500         | 0.500       | 0.501    |
| clean  | j1k3                    | 962  | 0.500         | 0.500       | 0.518    |
| clean  | j1k4                    | 366  | 0.500         | 0.500       | 0.516    |
| clean  | j1k5                    | 90   | 0.500         | 0.500       | 0.478    |
| clean  | j2k3                    | 1751 | 0.500         | 0.500       | 0.520    |
| clean  | j2k4                    | 911  | 0.502         | 0.499       | 0.502    |
| clean  | j2k5                    | 328  | 0.501         | 0.500       | 0.509    |
| clean  | j3k4                    | 1609 | 0.498         | 0.502       | 0.502    |
| clean  | j3k5                    | 911  | 0.499         | 0.499       | 0.508    |
| clean  | j4k5                    | 236  | 0.492         | 0.503       | 0.462    |
| clean  | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.507    |

## `traj_a1_s42` / venue `swap65k` / step 24000 / anchor: t_v (Bayesian-detectable violation)

Critic block `post_block7`.

| critic | a | n    | rank cons | rank incons | win rate | scalar rank |
|--------|---|------|-----------|-------------|----------|-------------|
| linear | 0 | 3629 | 0.493     | 0.527       | 0.438    | 0.515       |
| linear | 1 | 2452 | 0.499     | 0.507       | 0.496    | 0.502       |
| mlp    | 0 | 3629 | 0.513     | 0.558       | 0.440    | 0.537       |
| mlp    | 1 | 2452 | 0.511     | 0.553       | 0.440    | 0.536       |
| clean  | 0 | 3629 | 0.491     | 0.522       | 0.457    | 0.514       |
| clean  | 1 | 2452 | 0.532     | 0.516       | 0.534    | 0.519       |

Per-`j` breakdown (a = 0):

| critic | j                 | n    | rank cons | rank incons | win rate |
|--------|-------------------|------|-----------|-------------|----------|
| linear | j1                | 1512 | 0.517     | 0.519       | 0.480    |
| linear | j2                | 1295 | 0.521     | 0.533       | 0.465    |
| linear | j3                | 756  | 0.516     | 0.514       | 0.487    |
| linear | j4                | 66   | 0.518     | 0.511       | 0.500    |
| linear | **all j (n-wtd)** | 3629 | --        | --          | 0.476    |
| mlp    | j1                | 1512 | 0.531     | 0.544       | 0.488    |
| mlp    | j2                | 1295 | 0.524     | 0.537       | 0.478    |
| mlp    | j3                | 756  | 0.507     | 0.500       | 0.520    |
| mlp    | j4                | 66   | 0.475     | 0.468       | 0.500    |
| mlp    | **all j (n-wtd)** | 3629 | --        | --          | 0.491    |
| clean  | j1                | 1512 | 0.493     | 0.503       | 0.479    |
| clean  | j2                | 1295 | 0.513     | 0.530       | 0.472    |
| clean  | j3                | 756  | 0.527     | 0.531       | 0.491    |
| clean  | j4                | 66   | 0.527     | 0.513       | 0.470    |
| clean  | **all j (n-wtd)** | 3629 | --        | --          | 0.479    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 864  | 0.505         | 0.505       | 0.481    |
| linear | j1k3                    | 482  | 0.515         | 0.510       | 0.504    |
| linear | j1k4                    | 147  | 0.540         | 0.532       | 0.537    |
| linear | j2k3                    | 946  | 0.511         | 0.513       | 0.501    |
| linear | j2k4                    | 300  | 0.520         | 0.526       | 0.485    |
| linear | j2k5                    | 49   | 0.560         | 0.537       | 0.551    |
| linear | j3k4                    | 619  | 0.510         | 0.503       | 0.499    |
| linear | j3k5                    | 137  | 0.540         | 0.516       | 0.547    |
| linear | j4k5                    | 66   | 0.519         | 0.511       | 0.500    |
| linear | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.499    |
| mlp    | j1k2                    | 864  | 0.498         | 0.517       | 0.472    |
| mlp    | j1k3                    | 482  | 0.545         | 0.549       | 0.506    |
| mlp    | j1k4                    | 147  | 0.575         | 0.575       | 0.537    |
| mlp    | j2k3                    | 946  | 0.499         | 0.502       | 0.487    |
| mlp    | j2k4                    | 300  | 0.528         | 0.555       | 0.453    |
| mlp    | j2k5                    | 49   | 0.568         | 0.541       | 0.531    |
| mlp    | j3k4                    | 619  | 0.497         | 0.485       | 0.526    |
| mlp    | j3k5                    | 137  | 0.529         | 0.478       | 0.591    |
| mlp    | j4k5                    | 66   | 0.475         | 0.468       | 0.500    |
| mlp    | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.497    |
| clean  | j1k2                    | 864  | 0.497         | 0.502       | 0.484    |
| clean  | j1k3                    | 482  | 0.481         | 0.484       | 0.488    |
| clean  | j1k4                    | 147  | 0.506         | 0.505       | 0.551    |
| clean  | j2k3                    | 946  | 0.510         | 0.515       | 0.504    |
| clean  | j2k4                    | 300  | 0.505         | 0.524       | 0.485    |
| clean  | j2k5                    | 49   | 0.539         | 0.523       | 0.531    |
| clean  | j3k4                    | 619  | 0.517         | 0.514       | 0.507    |
| clean  | j3k5                    | 137  | 0.543         | 0.515       | 0.518    |
| clean  | j4k5                    | 66   | 0.526         | 0.513       | 0.470    |
| clean  | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.498    |

## `traj_a1_s42` / venue `swap65k` / step 64000 / anchor: edit onset (first_diff)

Critic block `post_block7`.

| critic | a | n     | rank cons | rank incons | win rate | scalar rank |
|--------|---|-------|-----------|-------------|----------|-------------|
| linear | 0 | 10308 | 0.492     | 0.493       | 0.507    | 0.493       |
| linear | 1 | 10126 | 0.497     | 0.503       | 0.499    | 0.497       |
| mlp    | 0 | 10308 | 0.492     | 0.502       | 0.496    | 0.498       |
| mlp    | 1 | 10126 | 0.495     | 0.502       | 0.495    | 0.497       |
| clean  | 0 | 10308 | 0.497     | 0.487       | 0.518    | 0.496       |
| clean  | 1 | 10126 | 0.519     | 0.497       | 0.534    | 0.506       |

Per-`j` breakdown (a = 0):

| critic | j                 | n     | rank cons | rank incons | win rate |
|--------|-------------------|-------|-----------|-------------|----------|
| linear | j1                | 3140  | 0.500     | 0.500       | 0.501    |
| linear | j2                | 3356  | 0.500     | 0.500       | 0.515    |
| linear | j3                | 3320  | 0.500     | 0.499       | 0.512    |
| linear | j4                | 492   | 0.470     | 0.469       | 0.496    |
| linear | **all j (n-wtd)** | 10308 | --        | --          | 0.509    |
| mlp    | j1                | 3140  | 0.500     | 0.500       | 0.490    |
| mlp    | j2                | 3356  | 0.499     | 0.500       | 0.506    |
| mlp    | j3                | 3320  | 0.500     | 0.499       | 0.519    |
| mlp    | j4                | 492   | 0.481     | 0.472       | 0.524    |
| mlp    | **all j (n-wtd)** | 10308 | --        | --          | 0.506    |
| clean  | j1                | 3140  | 0.500     | 0.500       | 0.492    |
| clean  | j2                | 3356  | 0.501     | 0.500       | 0.518    |
| clean  | j3                | 3320  | 0.500     | 0.499       | 0.519    |
| clean  | j4                | 492   | 0.475     | 0.468       | 0.506    |
| clean  | **all j (n-wtd)** | 10308 | --        | --          | 0.510    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 1542 | 0.500         | 0.500       | 0.492    |
| linear | j1k3                    | 962  | 0.500         | 0.500       | 0.507    |
| linear | j1k4                    | 366  | 0.500         | 0.500       | 0.505    |
| linear | j1k5                    | 90   | 0.500         | 0.500       | 0.500    |
| linear | j2k3                    | 1751 | 0.500         | 0.501       | 0.520    |
| linear | j2k4                    | 911  | 0.501         | 0.500       | 0.510    |
| linear | j2k5                    | 328  | 0.499         | 0.501       | 0.506    |
| linear | j3k4                    | 1609 | 0.500         | 0.499       | 0.520    |
| linear | j3k5                    | 911  | 0.500         | 0.498       | 0.504    |
| linear | j4k5                    | 236  | 0.481         | 0.490       | 0.479    |
| linear | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.509    |
| mlp    | j1k2                    | 1542 | 0.500         | 0.500       | 0.477    |
| mlp    | j1k3                    | 962  | 0.500         | 0.500       | 0.484    |
| mlp    | j1k4                    | 366  | 0.500         | 0.500       | 0.527    |
| mlp    | j1k5                    | 90   | 0.500         | 0.500       | 0.511    |
| mlp    | j2k3                    | 1751 | 0.499         | 0.500       | 0.499    |
| mlp    | j2k4                    | 911  | 0.501         | 0.500       | 0.507    |
| mlp    | j2k5                    | 328  | 0.500         | 0.500       | 0.524    |
| mlp    | j3k4                    | 1609 | 0.500         | 0.498       | 0.515    |
| mlp    | j3k5                    | 911  | 0.500         | 0.499       | 0.525    |
| mlp    | j4k5                    | 236  | 0.484         | 0.497       | 0.517    |
| mlp    | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.503    |
| clean  | j1k2                    | 1542 | 0.500         | 0.500       | 0.474    |
| clean  | j1k3                    | 962  | 0.500         | 0.500       | 0.500    |
| clean  | j1k4                    | 366  | 0.500         | 0.500       | 0.519    |
| clean  | j1k5                    | 90   | 0.500         | 0.500       | 0.533    |
| clean  | j2k3                    | 1751 | 0.500         | 0.501       | 0.522    |
| clean  | j2k4                    | 911  | 0.501         | 0.500       | 0.516    |
| clean  | j2k5                    | 328  | 0.500         | 0.500       | 0.506    |
| clean  | j3k4                    | 1609 | 0.500         | 0.499       | 0.520    |
| clean  | j3k5                    | 911  | 0.500         | 0.499       | 0.510    |
| clean  | j4k5                    | 236  | 0.496         | 0.494       | 0.504    |
| clean  | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.508    |

## `traj_a1_s42` / venue `swap65k` / step 64000 / anchor: t_v (Bayesian-detectable violation)

Critic block `post_block7`.

| critic | a | n    | rank cons | rank incons | win rate | scalar rank |
|--------|---|------|-----------|-------------|----------|-------------|
| linear | 0 | 3629 | 0.500     | 0.542       | 0.424    | 0.523       |
| linear | 1 | 2452 | 0.480     | 0.503       | 0.463    | 0.489       |
| mlp    | 0 | 3629 | 0.536     | 0.573       | 0.462    | 0.554       |
| mlp    | 1 | 2452 | 0.511     | 0.569       | 0.430    | 0.541       |
| clean  | 0 | 3629 | 0.482     | 0.525       | 0.426    | 0.511       |
| clean  | 1 | 2452 | 0.524     | 0.509       | 0.543    | 0.508       |

Per-`j` breakdown (a = 0):

| critic | j                 | n    | rank cons | rank incons | win rate |
|--------|-------------------|------|-----------|-------------|----------|
| linear | j1                | 1512 | 0.517     | 0.532       | 0.468    |
| linear | j2                | 1295 | 0.529     | 0.548       | 0.473    |
| linear | j3                | 756  | 0.511     | 0.528       | 0.456    |
| linear | j4                | 66   | 0.475     | 0.488       | 0.439    |
| linear | **all j (n-wtd)** | 3629 | --        | --          | 0.467    |
| mlp    | j1                | 1512 | 0.535     | 0.544       | 0.505    |
| mlp    | j2                | 1295 | 0.527     | 0.544       | 0.485    |
| mlp    | j3                | 756  | 0.517     | 0.523       | 0.493    |
| mlp    | j4                | 66   | 0.485     | 0.457       | 0.606    |
| mlp    | **all j (n-wtd)** | 3629 | --        | --          | 0.497    |
| clean  | j1                | 1512 | 0.482     | 0.511       | 0.429    |
| clean  | j2                | 1295 | 0.517     | 0.527       | 0.483    |
| clean  | j3                | 756  | 0.513     | 0.538       | 0.452    |
| clean  | j4                | 66   | 0.496     | 0.487       | 0.485    |
| clean  | **all j (n-wtd)** | 3629 | --        | --          | 0.454    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 864  | 0.505         | 0.518       | 0.465    |
| linear | j1k3                    | 482  | 0.514         | 0.533       | 0.463    |
| linear | j1k4                    | 147  | 0.542         | 0.547       | 0.524    |
| linear | j2k3                    | 946  | 0.519         | 0.526       | 0.489    |
| linear | j2k4                    | 300  | 0.535         | 0.555       | 0.473    |
| linear | j2k5                    | 49   | 0.519         | 0.517       | 0.510    |
| linear | j3k4                    | 619  | 0.508         | 0.507       | 0.498    |
| linear | j3k5                    | 137  | 0.514         | 0.527       | 0.511    |
| linear | j4k5                    | 66   | 0.475         | 0.487       | 0.455    |
| linear | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.482    |
| mlp    | j1k2                    | 864  | 0.506         | 0.513       | 0.489    |
| mlp    | j1k3                    | 482  | 0.549         | 0.559       | 0.506    |
| mlp    | j1k4                    | 147  | 0.556         | 0.577       | 0.503    |
| mlp    | j2k3                    | 946  | 0.499         | 0.505       | 0.505    |
| mlp    | j2k4                    | 300  | 0.535         | 0.561       | 0.475    |
| mlp    | j2k5                    | 49   | 0.583         | 0.581       | 0.490    |
| mlp    | j3k4                    | 619  | 0.499         | 0.493       | 0.519    |
| mlp    | j3k5                    | 137  | 0.537         | 0.540       | 0.489    |
| mlp    | j4k5                    | 66   | 0.484         | 0.456       | 0.606    |
| mlp    | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.502    |
| clean  | j1k2                    | 864  | 0.488         | 0.510       | 0.432    |
| clean  | j1k3                    | 482  | 0.472         | 0.503       | 0.434    |
| clean  | j1k4                    | 147  | 0.489         | 0.513       | 0.483    |
| clean  | j2k3                    | 946  | 0.520         | 0.517       | 0.494    |
| clean  | j2k4                    | 300  | 0.515         | 0.524       | 0.470    |
| clean  | j2k5                    | 49   | 0.496         | 0.480       | 0.510    |
| clean  | j3k4                    | 619  | 0.507         | 0.512       | 0.502    |
| clean  | j3k5                    | 137  | 0.514         | 0.535       | 0.474    |
| clean  | j4k5                    | 66   | 0.493         | 0.485       | 0.485    |
| clean  | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.469    |

## `traj_eps01_s42` / venue `a1` / step 64000 / anchor: edit onset (first_diff)

Critic block `post_block7`.

| critic | a | n    | rank cons | rank incons | win rate | scalar rank |
|--------|---|------|-----------|-------------|----------|-------------|
| linear | 0 | 1982 | 0.493     | 0.505       | 0.480    | 0.501       |
| linear | 1 | 1971 | 0.495     | 0.506       | 0.494    | 0.501       |
| mlp    | 0 | 1982 | 0.501     | 0.529       | 0.464    | 0.519       |
| mlp    | 1 | 1971 | 0.508     | 0.524       | 0.482    | 0.514       |
| clean  | 0 | 1982 | 0.495     | 0.500       | 0.492    | 0.499       |
| clean  | 1 | 1971 | 0.495     | 0.498       | 0.499    | 0.497       |

Per-`j` breakdown (a = 0):

| critic | j                 | n    | rank cons | rank incons | win rate |
|--------|-------------------|------|-----------|-------------|----------|
| linear | j1                | 371  | 0.542     | 0.549       | 0.501    |
| linear | j2                | 638  | 0.505     | 0.508       | 0.480    |
| linear | j3                | 662  | 0.496     | 0.504       | 0.502    |
| linear | j4                | 311  | 0.513     | 0.512       | 0.494    |
| linear | **all j (n-wtd)** | 1982 | --        | --          | 0.493    |
| mlp    | j1                | 371  | 0.551     | 0.583       | 0.431    |
| mlp    | j2                | 638  | 0.519     | 0.530       | 0.481    |
| mlp    | j3                | 662  | 0.494     | 0.510       | 0.480    |
| mlp    | j4                | 311  | 0.514     | 0.529       | 0.479    |
| mlp    | **all j (n-wtd)** | 1982 | --        | --          | 0.471    |
| clean  | j1                | 371  | 0.527     | 0.528       | 0.488    |
| clean  | j2                | 638  | 0.500     | 0.504       | 0.484    |
| clean  | j3                | 662  | 0.495     | 0.504       | 0.495    |
| clean  | j4                | 311  | 0.510     | 0.516       | 0.495    |
| clean  | **all j (n-wtd)** | 1982 | --        | --          | 0.490    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 195  | 0.500         | 0.500       | 0.508    |
| linear | j1k3                    | 97   | 0.500         | 0.500       | 0.479    |
| linear | j1k4                    | 46   | 0.500         | 0.500       | 0.478    |
| linear | j2k3                    | 229  | 0.503         | 0.500       | 0.487    |
| linear | j2k4                    | 110  | 0.502         | 0.500       | 0.505    |
| linear | j2k5                    | 30   | 0.497         | 0.500       | 0.483    |
| linear | j3k4                    | 209  | 0.503         | 0.495       | 0.550    |
| linear | j3k5                    | 123  | 0.499         | 0.501       | 0.467    |
| linear | j4k5                    | 22   | 0.498         | 0.472       | 0.591    |
| linear | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.504    |
| mlp    | j1k2                    | 195  | 0.500         | 0.500       | 0.477    |
| mlp    | j1k3                    | 97   | 0.500         | 0.500       | 0.505    |
| mlp    | j1k4                    | 46   | 0.500         | 0.500       | 0.478    |
| mlp    | j2k3                    | 229  | 0.500         | 0.501       | 0.513    |
| mlp    | j2k4                    | 110  | 0.502         | 0.500       | 0.509    |
| mlp    | j2k5                    | 30   | 0.505         | 0.493       | 0.450    |
| mlp    | j3k4                    | 209  | 0.501         | 0.497       | 0.493    |
| mlp    | j3k5                    | 123  | 0.500         | 0.501       | 0.528    |
| mlp    | j4k5                    | 22   | 0.515         | 0.474       | 0.545    |
| mlp    | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.500    |
| clean  | j1k2                    | 195  | 0.500         | 0.500       | 0.490    |
| clean  | j1k3                    | 97   | 0.500         | 0.500       | 0.495    |
| clean  | j1k4                    | 46   | 0.500         | 0.500       | 0.478    |
| clean  | j2k3                    | 229  | 0.503         | 0.499       | 0.500    |
| clean  | j2k4                    | 110  | 0.502         | 0.500       | 0.491    |
| clean  | j2k5                    | 30   | 0.500         | 0.497       | 0.467    |
| clean  | j3k4                    | 209  | 0.503         | 0.496       | 0.531    |
| clean  | j3k5                    | 123  | 0.499         | 0.501       | 0.463    |
| clean  | j4k5                    | 22   | 0.473         | 0.449       | 0.636    |
| clean  | **all (j, k*) (n-wtd)** | 1061 | coverage 0.54 | --          | 0.500    |

## `traj_eps01_s42` / venue `a1` / step 64000 / anchor: t_v (Bayesian-detectable violation)

Critic block `post_block7`.

| critic | a | n   | rank cons | rank incons | win rate | scalar rank |
|--------|---|-----|-----------|-------------|----------|-------------|
| linear | 0 | 465 | 0.468     | 0.535       | 0.396    | 0.504       |
| linear | 1 | 304 | 0.456     | 0.487       | 0.457    | 0.464       |
| mlp    | 0 | 465 | 0.517     | 0.567       | 0.434    | 0.536       |
| mlp    | 1 | 304 | 0.530     | 0.547       | 0.477    | 0.523       |
| clean  | 0 | 465 | 0.464     | 0.519       | 0.417    | 0.494       |
| clean  | 1 | 304 | 0.454     | 0.476       | 0.474    | 0.457       |

Per-`j` breakdown (a = 0):

| critic | j                 | n   | rank cons | rank incons | win rate |
|--------|-------------------|-----|-----------|-------------|----------|
| linear | j1                | 188 | 0.523     | 0.513       | 0.537    |
| linear | j2                | 176 | 0.504     | 0.500       | 0.523    |
| linear | j3                | 96  | 0.533     | 0.521       | 0.542    |
| linear | **all j (n-wtd)** | 460 | --        | --          | 0.533    |
| mlp    | j1                | 188 | 0.537     | 0.523       | 0.527    |
| mlp    | j2                | 176 | 0.521     | 0.531       | 0.443    |
| mlp    | j3                | 96  | 0.500     | 0.498       | 0.521    |
| mlp    | **all j (n-wtd)** | 460 | --        | --          | 0.493    |
| clean  | j1                | 188 | 0.516     | 0.506       | 0.516    |
| clean  | j2                | 176 | 0.504     | 0.492       | 0.500    |
| clean  | j3                | 96  | 0.535     | 0.523       | 0.510    |
| clean  | **all j (n-wtd)** | 460 | --        | --          | 0.509    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n   | rank cons     | rank incons | win rate |
|--------|-------------------------|-----|---------------|-------------|----------|
| linear | j1k2                    | 108 | 0.516         | 0.505       | 0.546    |
| linear | j1k3                    | 59  | 0.562         | 0.518       | 0.568    |
| linear | j2k3                    | 129 | 0.482         | 0.483       | 0.504    |
| linear | j2k4                    | 41  | 0.533         | 0.532       | 0.500    |
| linear | j3k4                    | 74  | 0.544         | 0.532       | 0.568    |
| linear | j3k5                    | 22  | 0.486         | 0.430       | 0.545    |
| linear | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.536    |
| mlp    | j1k2                    | 108 | 0.504         | 0.503       | 0.486    |
| mlp    | j1k3                    | 59  | 0.565         | 0.534       | 0.593    |
| mlp    | j2k3                    | 129 | 0.493         | 0.493       | 0.461    |
| mlp    | j2k4                    | 41  | 0.517         | 0.572       | 0.463    |
| mlp    | j3k4                    | 74  | 0.498         | 0.483       | 0.514    |
| mlp    | j3k5                    | 22  | 0.446         | 0.441       | 0.545    |
| mlp    | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.499    |
| clean  | j1k2                    | 108 | 0.511         | 0.506       | 0.528    |
| clean  | j1k3                    | 59  | 0.556         | 0.507       | 0.508    |
| clean  | j2k3                    | 129 | 0.486         | 0.485       | 0.473    |
| clean  | j2k4                    | 41  | 0.538         | 0.498       | 0.488    |
| clean  | j3k4                    | 74  | 0.543         | 0.530       | 0.554    |
| clean  | j3k5                    | 22  | 0.508         | 0.447       | 0.591    |
| clean  | **all (j, k*) (n-wtd)** | 433 | coverage 0.93 | --          | 0.513    |

## `traj_eps01_s42` / venue `swap65k` / step 64000 / anchor: edit onset (first_diff)

Critic block `post_block7`.

| critic | a | n     | rank cons | rank incons | win rate | scalar rank |
|--------|---|-------|-----------|-------------|----------|-------------|
| linear | 0 | 10308 | 0.475     | 0.494       | 0.477    | 0.487       |
| linear | 1 | 10126 | 0.492     | 0.495       | 0.505    | 0.491       |
| mlp    | 0 | 10308 | 0.486     | 0.508       | 0.470    | 0.497       |
| mlp    | 1 | 10126 | 0.497     | 0.504       | 0.484    | 0.495       |
| clean  | 0 | 10308 | 0.483     | 0.485       | 0.501    | 0.487       |
| clean  | 1 | 10126 | 0.498     | 0.485       | 0.528    | 0.491       |

Per-`j` breakdown (a = 0):

| critic | j                 | n     | rank cons | rank incons | win rate |
|--------|-------------------|-------|-----------|-------------|----------|
| linear | j1                | 3140  | 0.500     | 0.500       | 0.495    |
| linear | j2                | 3356  | 0.501     | 0.500       | 0.518    |
| linear | j3                | 3320  | 0.500     | 0.498       | 0.505    |
| linear | j4                | 492   | 0.472     | 0.469       | 0.504    |
| linear | **all j (n-wtd)** | 10308 | --        | --          | 0.506    |
| mlp    | j1                | 3140  | 0.500     | 0.500       | 0.494    |
| mlp    | j2                | 3356  | 0.500     | 0.500       | 0.510    |
| mlp    | j3                | 3320  | 0.500     | 0.499       | 0.508    |
| mlp    | j4                | 492   | 0.474     | 0.460       | 0.547    |
| mlp    | **all j (n-wtd)** | 10308 | --        | --          | 0.506    |
| clean  | j1                | 3140  | 0.500     | 0.500       | 0.503    |
| clean  | j2                | 3356  | 0.502     | 0.500       | 0.516    |
| clean  | j3                | 3320  | 0.500     | 0.498       | 0.506    |
| clean  | j4                | 492   | 0.468     | 0.463       | 0.502    |
| clean  | **all j (n-wtd)** | 10308 | --        | --          | 0.508    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 1542 | 0.500         | 0.500       | 0.492    |
| linear | j1k3                    | 962  | 0.500         | 0.500       | 0.492    |
| linear | j1k4                    | 366  | 0.500         | 0.500       | 0.505    |
| linear | j1k5                    | 90   | 0.500         | 0.500       | 0.533    |
| linear | j2k3                    | 1751 | 0.500         | 0.500       | 0.507    |
| linear | j2k4                    | 911  | 0.503         | 0.500       | 0.523    |
| linear | j2k5                    | 328  | 0.500         | 0.500       | 0.502    |
| linear | j3k4                    | 1609 | 0.500         | 0.498       | 0.513    |
| linear | j3k5                    | 911  | 0.501         | 0.498       | 0.508    |
| linear | j4k5                    | 236  | 0.500         | 0.503       | 0.479    |
| linear | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.505    |
| mlp    | j1k2                    | 1542 | 0.500         | 0.500       | 0.482    |
| mlp    | j1k3                    | 962  | 0.500         | 0.500       | 0.501    |
| mlp    | j1k4                    | 366  | 0.500         | 0.500       | 0.497    |
| mlp    | j1k5                    | 90   | 0.500         | 0.500       | 0.522    |
| mlp    | j2k3                    | 1751 | 0.500         | 0.500       | 0.506    |
| mlp    | j2k4                    | 911  | 0.500         | 0.500       | 0.525    |
| mlp    | j2k5                    | 328  | 0.500         | 0.500       | 0.509    |
| mlp    | j3k4                    | 1609 | 0.500         | 0.499       | 0.507    |
| mlp    | j3k5                    | 911  | 0.500         | 0.499       | 0.512    |
| mlp    | j4k5                    | 236  | 0.483         | 0.479       | 0.534    |
| mlp    | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.505    |
| clean  | j1k2                    | 1542 | 0.500         | 0.500       | 0.497    |
| clean  | j1k3                    | 962  | 0.500         | 0.500       | 0.502    |
| clean  | j1k4                    | 366  | 0.500         | 0.500       | 0.514    |
| clean  | j1k5                    | 90   | 0.500         | 0.500       | 0.533    |
| clean  | j2k3                    | 1751 | 0.501         | 0.500       | 0.506    |
| clean  | j2k4                    | 911  | 0.503         | 0.500       | 0.516    |
| clean  | j2k5                    | 328  | 0.501         | 0.500       | 0.494    |
| clean  | j3k4                    | 1609 | 0.501         | 0.498       | 0.503    |
| clean  | j3k5                    | 911  | 0.501         | 0.498       | 0.526    |
| clean  | j4k5                    | 236  | 0.498         | 0.497       | 0.475    |
| clean  | **all (j, k*) (n-wtd)** | 8706 | coverage 0.84 | --          | 0.506    |

## `traj_eps01_s42` / venue `swap65k` / step 64000 / anchor: t_v (Bayesian-detectable violation)

Critic block `post_block7`.

| critic | a | n    | rank cons | rank incons | win rate | scalar rank |
|--------|---|------|-----------|-------------|----------|-------------|
| linear | 0 | 3629 | 0.480     | 0.546       | 0.384    | 0.515       |
| linear | 1 | 2452 | 0.479     | 0.508       | 0.465    | 0.487       |
| mlp    | 0 | 3629 | 0.537     | 0.601       | 0.403    | 0.563       |
| mlp    | 1 | 2452 | 0.555     | 0.584       | 0.440    | 0.553       |
| clean  | 0 | 3629 | 0.464     | 0.522       | 0.414    | 0.497       |
| clean  | 1 | 2452 | 0.455     | 0.481       | 0.472    | 0.462       |

Per-`j` breakdown (a = 0):

| critic | j                 | n    | rank cons | rank incons | win rate |
|--------|-------------------|------|-----------|-------------|----------|
| linear | j1                | 1512 | 0.527     | 0.519       | 0.513    |
| linear | j2                | 1295 | 0.525     | 0.535       | 0.483    |
| linear | j3                | 756  | 0.536     | 0.524       | 0.532    |
| linear | j4                | 66   | 0.525     | 0.439       | 0.606    |
| linear | **all j (n-wtd)** | 3629 | --        | --          | 0.508    |
| mlp    | j1                | 1512 | 0.537     | 0.536       | 0.528    |
| mlp    | j2                | 1295 | 0.550     | 0.553       | 0.474    |
| mlp    | j3                | 756  | 0.545     | 0.555       | 0.466    |
| mlp    | j4                | 66   | 0.479     | 0.423       | 0.561    |
| mlp    | **all j (n-wtd)** | 3629 | --        | --          | 0.497    |
| clean  | j1                | 1512 | 0.507     | 0.505       | 0.493    |
| clean  | j2                | 1295 | 0.519     | 0.519       | 0.510    |
| clean  | j3                | 756  | 0.532     | 0.519       | 0.530    |
| clean  | j4                | 66   | 0.518     | 0.442       | 0.621    |
| clean  | **all j (n-wtd)** | 3629 | --        | --          | 0.509    |

Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an exact width-and-depth cell:

| critic | (j, k*)                 | n    | rank cons     | rank incons | win rate |
|--------|-------------------------|------|---------------|-------------|----------|
| linear | j1k2                    | 864  | 0.494         | 0.496       | 0.494    |
| linear | j1k3                    | 482  | 0.552         | 0.530       | 0.525    |
| linear | j1k4                    | 147  | 0.562         | 0.547       | 0.585    |
| linear | j2k3                    | 946  | 0.514         | 0.513       | 0.503    |
| linear | j2k4                    | 300  | 0.517         | 0.547       | 0.453    |
| linear | j2k5                    | 49   | 0.534         | 0.554       | 0.510    |
| linear | j3k4                    | 619  | 0.518         | 0.490       | 0.577    |
| linear | j3k5                    | 137  | 0.558         | 0.565       | 0.511    |
| linear | j4k5                    | 66   | 0.523         | 0.437       | 0.606    |
| linear | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.518    |
| mlp    | j1k2                    | 864  | 0.497         | 0.504       | 0.502    |
| mlp    | j1k3                    | 482  | 0.562         | 0.562       | 0.509    |
| mlp    | j1k4                    | 147  | 0.580         | 0.585       | 0.558    |
| mlp    | j2k3                    | 946  | 0.516         | 0.516       | 0.467    |
| mlp    | j2k4                    | 300  | 0.551         | 0.572       | 0.463    |
| mlp    | j2k5                    | 49   | 0.600         | 0.593       | 0.490    |
| mlp    | j3k4                    | 619  | 0.522         | 0.516       | 0.502    |
| mlp    | j3k5                    | 137  | 0.540         | 0.564       | 0.423    |
| mlp    | j4k5                    | 66   | 0.476         | 0.421       | 0.561    |
| mlp    | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.491    |
| clean  | j1k2                    | 864  | 0.488         | 0.493       | 0.477    |
| clean  | j1k3                    | 482  | 0.529         | 0.508       | 0.515    |
| clean  | j1k4                    | 147  | 0.537         | 0.519       | 0.592    |
| clean  | j2k3                    | 946  | 0.516         | 0.508       | 0.525    |
| clean  | j2k4                    | 300  | 0.508         | 0.528       | 0.463    |
| clean  | j2k5                    | 49   | 0.526         | 0.527       | 0.531    |
| clean  | j3k4                    | 619  | 0.516         | 0.485       | 0.569    |
| clean  | j3k5                    | 137  | 0.557         | 0.550       | 0.489    |
| clean  | j4k5                    | 66   | 0.517         | 0.442       | 0.606    |
| clean  | **all (j, k*) (n-wtd)** | 3610 | coverage 0.99 | --          | 0.517    |

