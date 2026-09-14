# rubato / kc2 — Phase 1: the ceiling of the factored program

seed 0 · R* 0.16 · tempi [32, 16, 8, 2] · Delta 5 (120 ms) · app=fixed · 1656s (library 581 · kin 1 · sweep 557)
complete=True

## Gates

| gate | what | result |
|---|---|---|
| P-F0 | the 13 donor constants | 13 checked, 0 bad — **pass** |
| P-T1 | with every new arm off, this fork reproduces `accelerando/t1` | 100 checks, max\|delta\| = 0.000e+00, applicable=True — **pass** |
| P-G | one pinned gain cell re-fit from the full grid | cell [8, 5]: re-fit [1.2, 0.12] vs pinned [1.2, 0.12], applicable=True — **pass** |
| P-TAU | tau measured from a step response | fast 30.00 ms vs 30.00; donor 499.9 vs 500.0 — **pass** |
| P-ZOH | the discretisation the oracle inverts is the plant's | alpha measured 0.44932909 vs analytic 0.44932896 (rel 2.9e-07) — **pass** |
| P-R | the command resampler is the identity at s = 1 | max\|delta\| = 0.000e+00 over 405 members — **pass** |
| P-N | the nesting op is exact at every tempo | 672 spelled, 0 bad, weld max\|delta\| = 6.02e-08 — **pass** |
| P-KN | the phase resampler splits at the span midpoint | max\|delta\| = 0.000e+00 over 672 members — **pass** |
| P-K | THE ORACLE IDENTITY, outside the rotation region | out max\|du\| = 9.373e-05 (tol 1e-03), rms 9.31e-07 over 77838 steps — **pass** |

**P-K, the residual against the world's own Gaussian gate weight.** The body model is the world model exactly where the command-rotation region is off; this is the table that says the residual is the rotation and nothing else.

| gate weight w | steps | rms \|du\| | max \|du\| |
|---|---|---|---|
| 0e+00 - 1e-06 | 71362 | 3.46e-08 | 1.79e-07 |
| 1e-06 - 1e-04 | 6476 | 3.23e-06 | 9.37e-05 |
| 1e-04 - 1e-02 | 9584 | 6.85e-04 | 7.89e-03 |
| 1e-02 - 1e+00 | 12802 | 1.37e-01 | 1.19e+00 |

In-region residual (REPORTED, not gated): rms 1.037e-01, max 1.193e+00 over 22386 steps.

**The two inverse schemes, algebraically.** Both are `u = c_a * a + c_v * v` and they agree exactly on drag (c_v = 0.200000 = 1/v_term). They differ only on acceleration: zoh 0.008717 against ct 0.006000, a ratio of **1.4528**, because dt_ctrl/tau = 0.800 is not small on this body.

**The forward step's one-step residual** (the same model, run forwards): clean world \|dv\| rms 2.321e-07 / max 5.331e-07; ROTATED world rms 2.624e-01 / max 2.233e+00, at a velocity scale of 0.159 m/s. The body model does not contain the world's command-rotation region, by construction.

## The piece at each tempo

| H | note ms | note/tau | speed m/s | u_turn (schedule) | band | do-nothing | reflex | reflex_ec | reflex_ec0 |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 768 | 25.60 | 0.16 | 0.04 | 0.0611 | 0.3603 | 0.0037 | 0.0252 | 0.0013 |
| 16 | 384 | 12.80 | 0.32 | 0.08 | 0.0611 | 0.3603 | 0.0103 | 0.0434 | 0.0027 |
| 8 | 192 | 6.40 | 0.64 | 0.17 | 0.0611 | 0.3603 | 0.0174 | 0.0661 | 0.0079 |
| 2 | 48 | 1.60 | 2.55 | 0.87 | 0.0611 | 0.3603 | 0.0690 | 0.1219 | 0.0320 |

`reflex_ec` / `reflex_ec0` are the REPORTED REFERENCE ROWS and are never headlined: the incumbent handed the same body model the learner holds, used only to carry its Delta-stale read forward along the commands it already issued (`accompanist` d3b's operator). `ec` keeps the Delta-fit gains; `ec0` plays the Delta = 0 gains an un-delayed read licenses.

## The sweep, e_piece (per-waypoint, drilled)

### `aud` — plant audition at the seam

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | kzoa_L1 | kzoa_L2 | kzoa_L3 | kzoa_L4 | kzsa_L1 | kzsa_L2 | kzsa_L3 | kzsa_L4 | kfl_L1 | kfl_L2 | kfl_L3 | kfl_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0037 | 0.0454* | 0.0266* | 0.0259* | 0.0263* | 0.0454* | 0.0266* | 0.0259* | 0.0263* | 0.0408* | 0.0399* | 0.0326* | 0.0213* | 0.0408* | 0.0399* | 0.0326* | 0.0213* | 0.0260* | 0.0297* | 0.0446* | 0.0429* |
| 16 | 0.0611 | 0.0103 | 0.0374* | 0.0401* | 0.0465* | 0.0093* | 0.0677 | 0.0623 | 0.0858 | 0.0951 | 0.0404* | 0.0253* | 0.0416* | 0.0271* | 0.0836 | 0.0426* | 0.0426* | 0.0446* | 0.0913 | 0.0460* | 0.1183 | 0.0795 |
| 8 | 0.0611 | 0.0174 | 0.0312* | 0.0289* | 0.0075* | 0.0123* | 0.1093 | 0.1412 | 0.1742 | 0.1461 | 0.0312* | 0.0243* | 0.0183* | 0.0226* | 0.0868 | 0.0838 | 0.0529* | 0.0550* | 0.0965 | 0.0520* | 0.0555* | 0.0663 |
| 2 | 0.0611 | 0.0690 | 0.0453* | 0.0298* | 0.0334* | 0.0069* | 0.2168 | 0.2323 | 0.2356 | 0.2452 | 0.0222* | 0.0343* | 0.0501* | 0.0578* | 0.3498 | 0.2033 | 0.1210 | 0.0660 | 0.4119 | 0.1735 | 0.1163 | 0.0893 |

### `key` — frozen posture key, no audition

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | kzoa_L1 | kzoa_L2 | kzoa_L3 | kzoa_L4 | kzsa_L1 | kzsa_L2 | kzsa_L3 | kzsa_L4 | kfl_L1 | kfl_L2 | kfl_L3 | kfl_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0037 | 0.0562* | 0.0842 | 0.0175* | 0.0175* | 0.0562* | 0.0842 | 0.0175* | 0.0175* | 0.0399* | 0.0872 | 0.0803 | 0.0839 | 0.0399* | 0.0872 | 0.0803 | 0.0839 | 0.0522* | 0.0997 | 0.0604* | 0.0627 |
| 16 | 0.0611 | 0.0103 | 0.0549* | 0.0188* | 0.0142* | 0.0087* | 0.0962 | 0.1087 | 0.0990 | 0.0990 | 0.0395* | 0.0370* | 0.0487* | 0.0287* | 0.0776 | 0.0339* | 0.0498* | 0.0719 | 0.1202 | 0.0694 | 0.0673 | 0.0821 |
| 8 | 0.0611 | 0.0174 | 0.0580* | 0.0544* | 0.0310* | 0.0185* | 0.1400 | 0.2349 | 0.2015 | 0.2015 | 0.0326* | 0.0288* | 0.0429* | 0.0348* | 0.1285 | 0.0545* | 0.0587* | 0.0498* | 0.1420 | 0.0564* | 0.0555* | 0.0574* |
| 2 | 0.0611 | 0.0690 | 0.0399* | 0.0181* | 0.0123* | 0.0074* | 0.2403 | 0.2234 | 0.2326 | 0.2326 | 0.0236* | 0.0398* | 0.0534* | 0.0542* | 0.2919 | 0.0846 | 0.1167 | 0.0938 | 0.2950 | 0.1260 | 0.1398 | 0.1160 |

`*` = inside the band.

**Pass fractions at three widths** (per-waypoint grade): fraction of the 32 shared eval geometries inside 1/2, 1/3 and 1/4 of a mean drilled leg. `aud` mode, deepest and shallowest rung plus the incumbent.

| H | arm | 1/2 leg | 1/3 leg | 1/4 leg |
|---|---|---|---|---|
| 32 | reflex | 1.00 | 1.00 | 1.00 |
| 32 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 32 | aud_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 32 | aud_s0_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_s0_L4 | 1.00 | 1.00 | 1.00 |
| 32 | aud_kzsa_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_kzsa_L2 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kzsa_L3 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kzsa_L4 | 1.00 | 1.00 | 1.00 |
| 16 | reflex | 1.00 | 1.00 | 1.00 |
| 16 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 16 | aud_L1 | 1.00 | 1.00 | 0.00 |
| 16 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 16 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 16 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzsa_L1 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzsa_L2 | 1.00 | 0.00 | 0.00 |
| 16 | aud_kzsa_L3 | 1.00 | 0.00 | 0.00 |
| 16 | aud_kzsa_L4 | 1.00 | 0.00 | 0.00 |
| 8 | reflex | 1.00 | 1.00 | 1.00 |
| 8 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 8 | aud_L1 | 1.00 | 1.00 | 0.00 |
| 8 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 8 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzsa_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzsa_L2 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzsa_L3 | 1.00 | 0.00 | 0.00 |
| 8 | aud_kzsa_L4 | 1.00 | 0.00 | 0.00 |
| 2 | reflex | 0.06 | 0.00 | 0.00 |
| 2 | reflex_ec0 | 1.00 | 1.00 | 0.00 |
| 2 | aud_L1 | 1.00 | 0.00 | 0.00 |
| 2 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 2 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L2 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L3 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L4 | 0.00 | 0.00 | 0.00 |

## The sweep, e_listen (the listener's clock)

### `aud` — plant audition at the seam

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | kzoa_L1 | kzoa_L2 | kzoa_L3 | kzoa_L4 | kzsa_L1 | kzsa_L2 | kzsa_L3 | kzsa_L4 | kfl_L1 | kfl_L2 | kfl_L3 | kfl_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0046 | 0.0435* | 0.0201* | 0.0247* | 0.0260* | 0.0435* | 0.0201* | 0.0247* | 0.0260* | 0.0416* | 0.0361* | 0.0308* | 0.0214* | 0.0416* | 0.0361* | 0.0308* | 0.0214* | 0.0292* | 0.0317* | 0.0387* | 0.0416* |
| 16 | 0.0611 | 0.0082 | 0.0330* | 0.0386* | 0.0433* | 0.0071* | 0.0599* | 0.0557* | 0.0792 | 0.0926 | 0.0390* | 0.0289* | 0.0417* | 0.0286* | 0.0784 | 0.0408* | 0.0422* | 0.0441* | 0.0868 | 0.0414* | 0.1038 | 0.0719 |
| 8 | 0.0611 | 0.0118 | 0.0300* | 0.0243* | 0.0077* | 0.0096* | 0.0966 | 0.1241 | 0.1533 | 0.1403 | 0.0305* | 0.0216* | 0.0177* | 0.0206* | 0.0785 | 0.0774 | 0.0485* | 0.0485* | 0.0870 | 0.0394* | 0.0476* | 0.0651 |
| 2 | 0.0611 | 0.0472 | 0.0359* | 0.0211* | 0.0308* | 0.0061* | 0.1509 | 0.1707 | 0.1802 | 0.1884 | 0.0193* | 0.0315* | 0.0490* | 0.0566* | 0.2553 | 0.1666 | 0.1141 | 0.0568* | 0.3036 | 0.1369 | 0.0986 | 0.0728 |

### `key` — frozen posture key, no audition

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | kzoa_L1 | kzoa_L2 | kzoa_L3 | kzoa_L4 | kzsa_L1 | kzsa_L2 | kzsa_L3 | kzsa_L4 | kfl_L1 | kfl_L2 | kfl_L3 | kfl_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0046 | 0.0434* | 0.0754 | 0.0205* | 0.0205* | 0.0434* | 0.0754 | 0.0205* | 0.0205* | 0.0361* | 0.0789 | 0.0754 | 0.0787 | 0.0361* | 0.0789 | 0.0754 | 0.0787 | 0.0483* | 0.0877 | 0.0542* | 0.0562* |
| 16 | 0.0611 | 0.0082 | 0.0450* | 0.0193* | 0.0151* | 0.0090* | 0.0849 | 0.0984 | 0.0924 | 0.0924 | 0.0324* | 0.0331* | 0.0467* | 0.0292* | 0.0753 | 0.0305* | 0.0425* | 0.0628 | 0.1169 | 0.0609* | 0.0577* | 0.0721 |
| 8 | 0.0611 | 0.0118 | 0.0521* | 0.0469* | 0.0331* | 0.0191* | 0.1280 | 0.2193 | 0.1855 | 0.1855 | 0.0287* | 0.0281* | 0.0419* | 0.0319* | 0.1203 | 0.0460* | 0.0541* | 0.0450* | 0.1348 | 0.0505* | 0.0529* | 0.0526* |
| 2 | 0.0611 | 0.0472 | 0.0339* | 0.0177* | 0.0094* | 0.0051* | 0.1939 | 0.1708 | 0.1794 | 0.1794 | 0.0209* | 0.0396* | 0.0529* | 0.0537* | 0.2518 | 0.0796 | 0.1095 | 0.0847 | 0.2493 | 0.1146 | 0.1292 | 0.1013 |

`*` = inside the band.

**Pass fractions at three widths** (listener grade): fraction of the 32 shared eval geometries inside 1/2, 1/3 and 1/4 of a mean drilled leg. `aud` mode, deepest and shallowest rung plus the incumbent.

| H | arm | 1/2 leg | 1/3 leg | 1/4 leg |
|---|---|---|---|---|
| 32 | reflex | 1.00 | 1.00 | 1.00 |
| 32 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 32 | aud_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 32 | aud_s0_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_s0_L4 | 1.00 | 1.00 | 1.00 |
| 32 | aud_kzsa_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_kzsa_L2 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kzsa_L3 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kzsa_L4 | 1.00 | 1.00 | 1.00 |
| 16 | reflex | 1.00 | 1.00 | 1.00 |
| 16 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 16 | aud_L1 | 1.00 | 1.00 | 0.00 |
| 16 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 16 | aud_s0_L1 | 1.00 | 0.00 | 0.00 |
| 16 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzsa_L1 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzsa_L2 | 1.00 | 0.00 | 0.00 |
| 16 | aud_kzsa_L3 | 1.00 | 0.00 | 0.00 |
| 16 | aud_kzsa_L4 | 1.00 | 0.00 | 0.00 |
| 8 | reflex | 1.00 | 1.00 | 1.00 |
| 8 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 8 | aud_L1 | 1.00 | 1.00 | 1.00 |
| 8 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 8 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzsa_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzsa_L2 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzsa_L3 | 1.00 | 0.00 | 0.00 |
| 8 | aud_kzsa_L4 | 1.00 | 0.00 | 0.00 |
| 2 | reflex | 1.00 | 0.06 | 0.00 |
| 2 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 2 | aud_L1 | 1.00 | 1.00 | 0.00 |
| 2 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 2 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L2 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L3 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L4 | 1.00 | 0.00 | 0.00 |

## The transfer question, isolated

Three numbers per (tempo, level), all `aud`, all in `e_piece`: the ORACLE content harvested at the asked tempo (`rec` — not deployable), the same-tempo kinematic conversion (`kzo`, s = 1 — the CONVERSION COST alone), and the kinematic unit carried from the slowest tempo (`kzs` — the transfer). `kzs / kzo` is what the tempo change costs once the conversion is paid for; `kzo / rec` is what the conversion costs.

| H | L | band | rec | kzo | kzs | scl0 | kzo/rec | kzs/kzo | kzs/scl0 | sat(kzs) |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.0611 | 0.0454* | — | — | 0.0454* | — | — | — | nan |
| 32 | 2 | 0.0611 | 0.0266* | — | — | 0.0266* | — | — | — | nan |
| 32 | 3 | 0.0611 | 0.0259* | — | — | 0.0259* | — | — | — | nan |
| 32 | 4 | 0.0611 | 0.0263* | — | — | 0.0263* | — | — | — | nan |
| 16 | 1 | 0.0611 | 0.0374* | — | — | 0.0677 | — | — | — | nan |
| 16 | 2 | 0.0611 | 0.0401* | — | — | 0.0623 | — | — | — | nan |
| 16 | 3 | 0.0611 | 0.0465* | — | — | 0.0858 | — | — | — | nan |
| 16 | 4 | 0.0611 | 0.0093* | — | — | 0.0951 | — | — | — | nan |
| 8 | 1 | 0.0611 | 0.0312* | — | — | 0.1093 | — | — | — | nan |
| 8 | 2 | 0.0611 | 0.0289* | — | — | 0.1412 | — | — | — | nan |
| 8 | 3 | 0.0611 | 0.0075* | — | — | 0.1742 | — | — | — | nan |
| 8 | 4 | 0.0611 | 0.0123* | — | — | 0.1461 | — | — | — | nan |
| 2 | 1 | 0.0611 | 0.0453* | — | — | 0.2168 | — | — | — | nan |
| 2 | 2 | 0.0611 | 0.0298* | — | — | 0.2323 | — | — | — | nan |
| 2 | 3 | 0.0611 | 0.0334* | — | — | 0.2356 | — | — | — | nan |
| 2 | 4 | 0.0611 | 0.0069* | — | — | 0.2452 | — | — | — | nan |

## The derivative scheme, measured

The exact (`zoh`) inverse against the continuous-time (`ct`) one the brief names, at the same content and the same source. They differ only in the coefficient on acceleration, by 1.4528x.

| H | L | kzo | kco | kco/kzo | kzs | kcs | kcs/kzs |
|---|---|---|---|---|---|---|---|
| 32 | 1 | — | — | — | — | — | — |
| 32 | 2 | — | — | — | — | — | — |
| 32 | 3 | — | — | — | — | — | — |
| 32 | 4 | — | — | — | — | — | — |
| 16 | 1 | — | — | — | — | — | — |
| 16 | 2 | — | — | — | — | — | — |
| 16 | 3 | — | — | — | — | — | — |
| 16 | 4 | — | — | — | — | — | — |
| 8 | 1 | — | — | — | — | — | — |
| 8 | 2 | — | — | — | — | — | — |
| 8 | 3 | — | — | — | — | — | — |
| 8 | 4 | — | — | — | — | — | — |
| 2 | 1 | — | — | — | — | — | — |
| 2 | 2 | — | — | — | — | — | — |
| 2 | 3 | — | — | — | — | — | — |
| 2 | 4 | — | — | — | — | — | — |

## Saturation: the actuator ceiling

Fraction of scalar command components the clip bound when the path was converted, per (arm, tempo, level), non-poison members only.

| arm | H32 L1 | H32 L2 | H32 L3 | H32 L4 | H16 L1 | H16 L2 | H16 L3 | H16 L4 | H8 L1 | H8 L2 | H8 L3 | H8 L4 | H2 L1 | H2 L2 | H2 L3 | H2 L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kfl | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kflc | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kzoa | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kzsa | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## The era table: which content wins at each tempo

Best arm at each tempo under each grade, over every committed arm (the incumbent and its two instruments excluded), and the same over the DEPLOYABLE arms only — those whose content exists without practising at the asked tempo (`s0`, `s2`, `kzs`, `kcs`).

| H | best (any) | e | in band | best DEPLOYABLE | e | in band | reflex | ratio |
|---|---|---|---|---|---|---|---|---|
| 32 | key_L3 | 0.0175 | yes | key_s0_L3 | 0.0175 | yes | 0.0037 | 0.21x |
| 16 | key_L4 | 0.0087 | yes | key_kzsa_L2 | 0.0339 | yes | 0.0103 | 0.30x |
| 8 | aud_L3 | 0.0075 | yes | key_kflc_L4 | 0.0498 | yes | 0.0174 | 0.35x |
| 2 | aud_L4 | 0.0069 | yes | aud_kflc_L4 | 0.0660 | no | 0.0690 | 1.05x |

## Parity per slot against the same-tempo recording

A slot is OPEN at a tempo iff the arm's content, EXECUTED ON THE PLANT from that slot's HELD-OUT launch states, is at least as good as the same-tempo recording's own audition winner on at least tau = 0.75 of them. `solo`'s gate with the tempo added to the index; the reference is the per-state winner, not a fixed member. The reference side is an instrument and is not charged.

| arm | H32 L1 | H32 L2 | H32 L3 | H32 L4 | H16 L1 | H16 L2 | H16 L3 | H16 L4 | H8 L1 | H8 L2 | H8 L3 | H8 L4 | H2 L1 | H2 L2 | H2 L3 | H2 L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Mean executed error on the held-out states, arm vs the recording it is measured against (arm / reference):

| arm | H32 L1 | H32 L2 | H32 L3 | H32 L4 | H16 L1 | H16 L2 | H16 L3 | H16 L4 | H8 L1 | H8 L2 | H8 L3 | H8 L4 | H2 L1 | H2 L2 | H2 L3 | H2 L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Phase 2 — the learned cerebellum

**The linear family's recovered coefficients**, against the analytic inverse it is trying to be. Fitted as a general 2x4 map with a bias, so the isotropy and the which-channel-depends-on-what were NOT given to it.

| | fitted (x, y) | analytic |
|---|---|---|
| drag `c_v` | 0.168617, 0.159768 | 0.200000 |
| accel `c_a` | 0.008360, 0.008296 | 0.008717 |
| off-diagonal (max abs) | 1.84e-02 | 0 |
| bias (max abs) | 3.58e-03 | 0 |

**The clean-world discriminator (decision 34)** — the same renditions replayed with the rotation region OFF, linear family refitted:

| fit | c_v | c_a | off-diag max | bias max |
|---|---|---|---|---|
| rotated world | 0.168617, 0.159768 | 0.008360, 0.008296 | 1.84e-02 | 3.58e-03 |
| **clean world** | 0.200000, 0.200000 | 0.008717, 0.008717 | 9.90e-11 | 3.68e-11 |

**Held-out fit at every tempo, in command units**, beside how far outside the training region the query lands. `reach` is the query's 99.5th-percentile magnitude over the fit's; `frac out` is the fraction of query points beyond the fit's 99.5th percentile. A faster tempo scales `|v|` by `s` and `|a|` by `s^2`, so this is the extrapolation question asked directly.

| H | fitted? | lin mse | mlp mse | v reach | a reach | v frac out | a frac out |
|---|---|---|---|---|---|---|---|
| 32 | yes | 4.786e-03 | 2.385e-03 | 1.09x | 1.10x | 0.004 | 0.009 |
| 16 | yes | 1.243e-03 | 1.507e-03 | 1.55x | 0.88x | 0.008 | 0.000 |
| 8 | no | 1.158e-03 | 1.268e-03 | 1.00x | 0.88x | 0.000 | 0.000 |
| 2 | no | 1.262e-02 | 2.390e-02 | 1.53x | 1.19x | 0.328 | 0.109 |

**How much of the ceiling's in-band set survives.** `kzsa` is the oracle on the same resampler (decision 22); `kfl` and `kfm` are the two fitted families on exactly that resampler, so the only difference is where the inverse came from. `aud`, `e_piece`.

| H | L | band | kzsa (ceiling) | kfl (linear) | kfm (MLP) | kfl/kzsa | kfm/kzsa |
|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.0611 | 0.0408* | 0.0260* | — | 0.64x | — |
| 32 | 2 | 0.0611 | 0.0399* | 0.0297* | — | 0.74x | — |
| 32 | 3 | 0.0611 | 0.0326* | 0.0446* | — | 1.37x | — |
| 32 | 4 | 0.0611 | 0.0213* | 0.0429* | — | 2.01x | — |
| 16 | 1 | 0.0611 | 0.0836 | 0.0913 | — | 1.09x | — |
| 16 | 2 | 0.0611 | 0.0426* | 0.0460* | — | 1.08x | — |
| 16 | 3 | 0.0611 | 0.0426* | 0.1183 | — | 2.78x | — |
| 16 | 4 | 0.0611 | 0.0446* | 0.0795 | — | 1.78x | — |
| 8 | 1 | 0.0611 | 0.0868 | 0.0965 | — | 1.11x | — |
| 8 | 2 | 0.0611 | 0.0838 | 0.0520* | — | 0.62x | — |
| 8 | 3 | 0.0611 | 0.0529* | 0.0555* | — | 1.05x | — |
| 8 | 4 | 0.0611 | 0.0550* | 0.0663 | — | 1.20x | — |
| 2 | 1 | 0.0611 | 0.3498 | 0.4119 | — | 1.18x | — |
| 2 | 2 | 0.0611 | 0.2033 | 0.1735 | — | 0.85x | — |
| 2 | 3 | 0.0611 | 0.1210 | 0.1163 | — | 0.96x | — |
| 2 | 4 | 0.0611 | 0.0660 | 0.0893 | — | 1.35x | — |

**In-band cells: ceiling 9, linear 7, MLP 0** (of 16).

## Phase 3 — the corrected executor

Feedforward from the kinematic program, corrected online against a forecast rolled from the Delta-stale read through the commands this executor already issued. `R` is the read period in control steps; **every read is charged as one feedback event**, the launch read included.

### The PD gains (decision 36: eigenvalues of the exact discrete error system placed at ρ), per tempo, with edges and the R-invariance verdict

| H | ρ\* (R = 1) | kp | kd | e_piece | edge | ρ\* at R = 16 | R-invariant? |
|---|---|---|---|---|---|---|---|
| 32 | 0.8 | 0.6053 | -0.062945 | 0.0027 | no | 0.6 | **no** |
| 16 | 0.3 | 7.415 | 0.20775 | 0.0119 | no | 0 | **no** |
| 8 | 0 | 15.13 | 0.32083 | 0.0188 | **EDGE** | 0.8 | **no** |
| 2 | 0.7 | 1.362 | -0.00058401 | 0.1671 | no | 0.7 | yes |

A negative `kd` is the allowed-and-reported branch of the rule (decision 36), not a clip. The frozen table is the R = 1 fit.

### (a) The fewest reads per span at which the corrected unit is in band

`x` = never in band at any read count on the ladder. **The ladder is capped at R = 16 control steps**, so at the slow tempi its coarsest rung is still many reads per span (16 at H = 32 L4) — those rows are the LADDER's floor, not a measured minimum. The true one-read points are the open-loop columns.

| H | L | exact FM | fitted FM | open `kfl` (1 read) | open `kzsa` (1 read) |
|---|---|---|---|---|---|
| 32 | 1 | **1** (R=32) 0.0434 | **1** (R=32) 0.0437 | 0.0522* | 0.0399* |
| 32 | 2 | **1** (R=64) 0.0497 | **1** (R=64) 0.0566 | 0.0997 | 0.0872 |
| 32 | 3 | **1** (R=128) 0.0368 | **4** (R=32) 0.0459 | 0.0604* | 0.0803 |
| 32 | 4 | **1** (R=256) 0.0239 | **1** (R=256) 0.0578 | 0.0627 | 0.0839 |
| 16 | 1 | **1** (R=16) 0.0540 | **1** (R=16) 0.0530 | 0.1202 | 0.0776 |
| 16 | 2 | **1** (R=32) 0.0603 | **1** (R=32) 0.0459 | 0.0694 | 0.0339* |
| 16 | 3 | **1** (R=64) 0.0568 | **4** (R=16) 0.0564 | 0.0673 | 0.0498* |
| 16 | 4 | **2** (R=64) 0.0574 | **16** (R=8) 0.0349 | 0.0821 | 0.0719 |
| 8 | 1 | **2** (R=4) 0.0531 | **4** (R=2) 0.0360 | 0.1420 | 0.1285 |
| 8 | 2 | **4** (R=4) 0.0354 | **1** (R=16) 0.0570 | 0.0564* | 0.0545* |
| 8 | 3 | **4** (R=8) 0.0542 | **4** (R=8) 0.0590 | 0.0555* | 0.0587* |
| 8 | 4 | **4** (R=16) 0.0496 | **4** (R=16) 0.0599 | 0.0574* | 0.0498* |
| 2 | 1 | x | x | 0.2950 | 0.2919 |
| 2 | 2 | x | x | 0.1260 | 0.0846 |
| 2 | 3 | x | x | 0.1398 | 0.1167 |
| 2 | 4 | x | x | 0.1160 | 0.0938 |

### The full read ladder (exact FM)

| H | L | rows as `e_piece` [realised reads per span], coarsest first |
|---|---|---|
| 32 | 1 | 0.0434* [1]  0.0160* [2]  0.0158* [4]  0.0189* [8]  0.0189* [16] |
| 32 | 2 | 0.0497* [1]  0.0493* [2]  0.0443* [4]  0.0431* [8]  0.0434* [16] |
| 32 | 3 | 0.0368* [1]  0.0369* [2]  0.0374* [4]  0.0382* [8]  0.0406* [16] |
| 32 | 4 | 0.0239* [1]  0.0210* [2]  0.0236* [4]  0.0226* [8]  0.0215* [16] |
| 16 | 1 | 0.0540* [1]  0.0198* [2]  0.0215* [4]  0.0162* [8]  0.0203* [16] |
| 16 | 2 | 0.0603* [1]  0.0617 [2]  0.0447* [4]  0.0459* [8]  0.0440* [16] |
| 16 | 3 | 0.0568* [1]  0.0606* [2]  0.0571* [4]  0.0448* [8]  0.0449* [16] |
| 16 | 4 | 0.0760 [1]  0.0574* [2]  0.0375* [4]  0.0354* [8]  0.0303* [16] |
| 8 | 1 | 0.0727 [1]  0.0531* [2]  0.0284* [4]  0.0204* [8] |
| 8 | 2 | 0.0745 [1]  0.0696 [2]  0.0354* [4]  0.0288* [8]  0.0288* [16] |
| 8 | 3 | 0.0811 [1]  0.0628 [2]  0.0542* [4]  0.0413* [8]  0.0394* [16] |
| 8 | 4 | 0.0979 [1]  0.0714 [2]  0.0496* [4]  0.0371* [8]  0.0252* [16] |
| 2 | 1 | 0.1704 [1]  0.1598 [2] |
| 2 | 2 | 0.1629 [1]  0.1325 [2]  0.1211 [4] |
| 2 | 3 | 0.1778 [1]  0.1459 [2]  0.1292 [4]  0.1206 [8] |
| 2 | 4 | 0.2566 [1]  0.1822 [2]  0.1343 [4]  0.1160 [8]  0.1073 [16] |

### (c) `reflex_ec` beside the R = 1 row, and the oracle-feedforward control

| H | reflex (fb) | reflex_ec | reflex_ec0 | cor densest L1 (fb) | L4 (fb) | oracle-ff densest L1 | L4 |
|---|---|---|---|---|---|---|---|
| 32 | 0.0037 (256) | 0.0252 | 0.0013 | 0.0189 (128) | 0.0215 (16) | 0.0052 | 0.0219 |
| 16 | 0.0103 (128) | 0.0434 | 0.0027 | 0.0203 (128) | 0.0303 (16) | 0.0207 | 0.0307 |
| 8 | 0.0174 (64) | 0.0661 | 0.0079 | 0.0204 (64) | 0.0252 (16) | 0.0196 | 0.0253 |
| 2 | 0.0690 (16) | 0.1219 | 0.0320 | 0.1598 (16) | 0.1073 (16) | 0.1670 | 0.0904 |

### (b) P-E, with and without correction

`e_piece` across ε = [0.0, 0.02, 0.05, 0.2] added to the FEEDFORWARD command only. `open` is the same executor with no reads past launch, so the two sides differ in exactly the correction. Beneath, the ε at which the 0.0611 band is spent, linearly interpolated on the ladder.

| H | L | rows by reads per span: `e_piece` across ε |
|---|---|---|
| 32 | 1 | n=1: 0.0434 / 0.0427 / 0.0472 / 0.1299  ·  n=4: 0.0158 / 0.0197 / 0.0293 / 0.0963  ·  n=16: 0.0189 / 0.0180 / 0.0281 / 0.0952 |
| 32 | 4 | n=1: 0.0239 / 0.0299 / 0.0455 / 0.1129  ·  n=4: 0.0236 / 0.0278 / 0.0372 / 0.1025  ·  n=16: 0.0215 / 0.0243 / 0.0347 / 0.0971 |
| 16 | 1 | n=1: 0.0540 / 0.0541 / 0.0554 / 0.0678  ·  n=4: 0.0215 / 0.0217 / 0.0204 / 0.0236  ·  n=16: 0.0203 / 0.0222 / 0.0200 / 0.0232 |
| 16 | 4 | n=1: 0.0760 / 0.0757 / 0.0754 / 0.0751  ·  n=4: 0.0375 / 0.0377 / 0.0385 / 0.0467  ·  n=16: 0.0303 / 0.0304 / 0.0306 / 0.0344 |
| 8 | 1 | n=1: 0.0727 / 0.0712 / 0.0736 / 0.0898  ·  n=4: 0.0284 / 0.0231 / 0.0223 / 0.0283  ·  n=8: 0.0204 / 0.0213 / 0.0222 / 0.0247 |
| 8 | 4 | n=1: 0.0979 / 0.0980 / 0.0986 / 0.1059  ·  n=4: 0.0496 / 0.0498 / 0.0503 / 0.0568  ·  n=16: 0.0252 / 0.0252 / 0.0253 / 0.0284 |
| 2 | 1 | n=1: 0.1704 / 0.1716 / 0.1759 / 0.2086  ·  n=2: 0.1598 / 0.1618 / 0.1897 / 0.1942 |
| 2 | 4 | n=1: 0.2566 / 0.2570 / 0.2584 / 0.2567  ·  n=4: 0.1343 / 0.1347 / 0.1343 / 0.1376  ·  n=16: 0.1073 / 0.1071 / 0.1071 / 0.1138 |

**ε at which the band is spent** (`n = 1` is launch only, i.e. uncorrected):

| H | L | by reads per span |
|---|---|---|
| 32 | 1 | n=1: 0.075  ·  n=4: 0.121  ·  n=16: 0.124 |
| 32 | 4 | n=1: 0.085  ·  n=4: 0.105  ·  n=16: 0.113 |
| 16 | 1 | n=1: 0.119  ·  n=4: >0.2  ·  n=16: >0.2 |
| 16 | 4 | n=1: already  ·  n=4: >0.2  ·  n=16: >0.2 |
| 8 | 1 | n=1: already  ·  n=4: >0.2  ·  n=8: >0.2 |
| 8 | 4 | n=1: already  ·  n=4: >0.2  ·  n=16: >0.2 |
| 2 | 1 | n=1: already  ·  n=2: already |
| 2 | 4 | n=1: already  ·  n=4: already  ·  n=16: already |

## The stale hand-over per seam (decision 23)

A level-l arm re-decides at `K / 2^(l-1)` drilled seams — 8 / 4 / 2 / 1 at L1 / L2 / L3 / L4 — and every one is a launch from a Delta-old read. `read_dx` / `read_dv` are the read's own error at that seam; `e_seg` is the arm's realised waypoint error there. NOTE, against what this table's first draft asserted: the read error is NOT arm-independent. It is the distance the body travels in Delta steps, so it depends on where the arm has put the body and how fast it is going, and `klsmoke1` measures it differing by 1.6x BETWEEN ARMS at the same tempo (H = 2: L1 0.3780, L2 0.3053, L4 0.2352). Read it as a joint property of the delay, the tempo and the arm.

### `aud_kzsa_L1`

| H | launches | mean read_dx | mean read_dv | e_seg by drilled seam |
|---|---|---|---|---|
| 32 | 8 | 0.0186 | 0.828 | 0.016 0.060 0.028 0.025 0.076 0.033 0.052 0.036 |
| 16 | 8 | 0.0432 | 1.283 | 0.026 0.091 0.074 0.063 0.089 0.087 0.102 0.137 |
| 8 | 8 | 0.0907 | 1.320 | 0.065 0.105 0.119 0.085 0.109 0.016 0.083 0.112 |
| 2 | 8 | 0.3780 | 3.976 | 0.111 0.296 0.414 0.484 0.518 0.502 0.354 0.118 |

### `aud_kzsa_L2`

| H | launches | mean read_dx | mean read_dv | e_seg by drilled seam |
|---|---|---|---|---|
| 32 | 4 | 0.0192 | 0.869 | 0.005 0.009 0.065 0.088 0.027 0.056 0.039 0.028 |
| 16 | 4 | 0.0454 | 1.136 | 0.021 0.025 0.030 0.037 0.060 0.054 0.054 0.060 |
| 8 | 4 | 0.0910 | 1.300 | 0.035 0.026 0.065 0.094 0.147 0.128 0.101 0.075 |
| 2 | 4 | 0.3053 | 4.165 | 0.073 0.046 0.074 0.082 0.248 0.371 0.395 0.337 |

### `aud_kzsa_L4`

| H | launches | mean read_dx | mean read_dv | e_seg by drilled seam |
|---|---|---|---|---|
| 32 | 1 | 0.0207 | 0.929 | 0.010 0.006 0.020 0.011 0.056 0.001 0.039 0.026 |
| 16 | 1 | 0.0347 | 1.095 | 0.021 0.025 0.025 0.057 0.091 0.041 0.051 0.046 |
| 8 | 1 | 0.0711 | 0.913 | 0.021 0.026 0.021 0.075 0.104 0.064 0.061 0.068 |
| 2 | 1 | 0.2352 | 3.547 | 0.042 0.063 0.089 0.111 0.072 0.043 0.041 0.067 |

## The ledger, drilled only (per performer)

| H | arm | n_fb | n_ground | t_piece (s) |
|---|---|---|---|---|
| 32 | reflex | 256.0 | 0.0 | 31.744 |
| 32 | reflex_ec | 256.0 | 0.0 | 31.744 |
| 32 | aud_L4 | 1.0 | 24.0 | 156.100 |
| 32 | key_L4 | 1.0 | 0.0 | 6.244 |
| 16 | reflex | 128.0 | 0.0 | 15.872 |
| 16 | reflex_ec | 128.0 | 0.0 | 15.872 |
| 16 | aud_L4 | 1.0 | 24.0 | 79.300 |
| 16 | key_L4 | 1.0 | 0.0 | 3.172 |
| 8 | reflex | 64.0 | 0.0 | 7.936 |
| 8 | reflex_ec | 64.0 | 0.0 | 7.936 |
| 8 | aud_L4 | 1.0 | 24.0 | 40.900 |
| 8 | key_L4 | 1.0 | 0.0 | 1.636 |
| 2 | reflex | 16.0 | 0.0 | 1.984 |
| 2 | reflex_ec | 16.0 | 0.0 | 1.984 |
| 2 | aud_L4 | 1.0 | 24.0 | 12.100 |
| 2 | key_L4 | 1.0 | 0.0 | 0.484 |

