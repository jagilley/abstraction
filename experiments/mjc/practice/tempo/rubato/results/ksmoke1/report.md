# rubato / ksmoke1 — Phase 1: the ceiling of the factored program

seed 0 · R* 0.16 · tempi [32, 8, 2] · Delta 5 (120 ms) · app=fixed · 825s (library 362 · kin 0 · sweep 353)
complete=True

## Gates

| gate | what | result |
|---|---|---|
| P-F0 | the 13 donor constants | 13 checked, 0 bad — **pass** |
| P-T1 | with every new arm off, this fork reproduces `accelerando/t1` | 75 checks, max\|delta\| = 0.000e+00, applicable=True — **pass** |
| P-G | one pinned gain cell re-fit from the full grid | cell [8, 5]: re-fit [1.2, 0.12] vs pinned [1.2, 0.12], applicable=True — **pass** |
| P-TAU | tau measured from a step response | fast 30.00 ms vs 30.00; donor 499.9 vs 500.0 — **pass** |
| P-ZOH | the discretisation the oracle inverts is the plant's | alpha measured 0.44932909 vs analytic 0.44932896 (rel 2.9e-07) — **pass** |
| P-R | the command resampler is the identity at s = 1 | max\|delta\| = 0.000e+00 over 405 members — **pass** |
| P-N | the nesting op is exact at every tempo | 504 spelled, 0 bad, weld max\|delta\| = 6.02e-08 — **pass** |
| P-KN | the phase resampler splits at the span midpoint | max\|delta\| = 0.000e+00 over 504 members — **pass** |
| P-K | THE ORACLE IDENTITY, outside the rotation region | out max\|du\| = 9.373e-05 (tol 1e-03), rms 1.09e-06 over 56652 steps — **pass** |

**P-K, the residual against the world's own Gaussian gate weight.** The body model is the world model exactly where the command-rotation region is off; this is the table that says the residual is the rotation and nothing else.

| gate weight w | steps | rms \|du\| | max \|du\| |
|---|---|---|---|
| 0e+00 - 1e-06 | 52080 | 3.92e-08 | 1.79e-07 |
| 1e-06 - 1e-04 | 4572 | 3.84e-06 | 9.37e-05 |
| 1e-04 - 1e-02 | 6958 | 7.80e-04 | 7.89e-03 |
| 1e-02 - 1e+00 | 8966 | 1.58e-01 | 1.19e+00 |

In-region residual (REPORTED, not gated): rms 1.186e-01, max 1.193e+00 over 15924 steps.

**The two inverse schemes, algebraically.** Both are `u = c_a * a + c_v * v` and they agree exactly on drag (c_v = 0.200000 = 1/v_term). They differ only on acceleration: zoh 0.008717 against ct 0.006000, a ratio of **1.4528**, because dt_ctrl/tau = 0.800 is not small on this body.

**The forward step's one-step residual** (the same model, run forwards): clean world \|dv\| rms 2.321e-07 / max 5.331e-07; ROTATED world rms 2.624e-01 / max 2.233e+00, at a velocity scale of 0.159 m/s. The body model does not contain the world's command-rotation region, by construction.

## The piece at each tempo

| H | note ms | note/tau | speed m/s | u_turn (schedule) | band | do-nothing | reflex | reflex_ec | reflex_ec0 |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 768 | 25.60 | 0.16 | 0.04 | 0.0611 | 0.3603 | 0.0037 | 0.0252 | 0.0013 |
| 8 | 192 | 6.40 | 0.64 | 0.17 | 0.0611 | 0.3603 | 0.0174 | 0.0661 | 0.0079 |
| 2 | 48 | 1.60 | 2.55 | 0.87 | 0.0611 | 0.3603 | 0.0690 | 0.1219 | 0.0320 |

`reflex_ec` / `reflex_ec0` are the REPORTED REFERENCE ROWS and are never headlined: the incumbent handed the same body model the learner holds, used only to carry its Delta-stale read forward along the commands it already issued (`accompanist` d3b's operator). `ec` keeps the Delta-fit gains; `ec0` plays the Delta = 0 gains an un-delayed read licenses.

## The sweep, e_piece (per-waypoint, drilled)

### `aud` — plant audition at the seam

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | s2_L1 | s2_L2 | s2_L3 | s2_L4 | kzs_L1 | kzs_L2 | kzs_L3 | kzs_L4 | kzo_L1 | kzo_L2 | kzo_L3 | kzo_L4 | kcs_L1 | kcs_L2 | kcs_L3 | kcs_L4 | kco_L1 | kco_L2 | kco_L3 | kco_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0037 | 0.0454* | 0.0266* | 0.0259* | 0.0263* | 0.0454* | 0.0266* | 0.0259* | 0.0263* | 0.0454* | 0.0266* | 0.0259* | 0.0263* | 0.0408* | 0.0399* | 0.0326* | 0.0213* | 0.0408* | 0.0399* | 0.0326* | 0.0213* | 0.0313* | 0.0272* | 0.0249* | 0.0298* | 0.0313* | 0.0272* | 0.0249* | 0.0298* |
| 8 | 0.0611 | 0.0174 | 0.0312* | 0.0289* | 0.0075* | 0.0123* | 0.1093 | 0.1412 | 0.1742 | 0.1461 | 0.3095 | 0.3082 | 0.3470 | 0.4290 | 0.1611 | 0.2340 | 0.3144 | 0.2944 | 0.0312* | 0.0243* | 0.0183* | 0.0226* | 0.1059 | 0.1525 | 0.2958 | 0.2873 | 0.0333* | 0.0143* | 0.0164* | 0.0261* |
| 2 | 0.0611 | 0.0690 | 0.0453* | 0.0298* | 0.0334* | 0.0069* | 0.2168 | 0.2323 | 0.2356 | 0.2452 | 0.4944 | 0.4081 | 0.3386 | 0.4132 | 0.3999 | 0.2626 | 0.1968 | 0.3184 | 0.0222* | 0.0343* | 0.0501* | 0.0578* | 0.4369 | 0.2827 | 0.2311 | 0.2850 | 0.0508* | 0.0674 | 0.0907 | 0.0750 |

### `key` — frozen posture key, no audition

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | s2_L1 | s2_L2 | s2_L3 | s2_L4 | kzs_L1 | kzs_L2 | kzs_L3 | kzs_L4 | kzo_L1 | kzo_L2 | kzo_L3 | kzo_L4 | kcs_L1 | kcs_L2 | kcs_L3 | kcs_L4 | kco_L1 | kco_L2 | kco_L3 | kco_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0037 | 0.0562* | 0.0842 | 0.0175* | 0.0175* | 0.0562* | 0.0842 | 0.0175* | 0.0175* | 0.0562* | 0.0842 | 0.0175* | 0.0175* | 0.0399* | 0.0872 | 0.0803 | 0.0839 | 0.0399* | 0.0872 | 0.0803 | 0.0839 | 0.0523* | 0.0683 | 0.0398* | 0.0434* | 0.0523* | 0.0683 | 0.0398* | 0.0434* |
| 8 | 0.0611 | 0.0174 | 0.0580* | 0.0544* | 0.0310* | 0.0185* | 0.1400 | 0.2349 | 0.2015 | 0.2015 | 0.9708 | 0.5232 | 0.8218 | 0.9555 | 0.5547 | 0.8980 | 0.4709 | 0.4651 | 0.0326* | 0.0288* | 0.0429* | 0.0348* | 0.6774 | 0.9637 | 0.4540 | 0.5145 | 0.0249* | 0.0095* | 0.0404* | 0.0296* |
| 2 | 0.0611 | 0.0690 | 0.0399* | 0.0181* | 0.0123* | 0.0074* | 0.2403 | 0.2234 | 0.2326 | 0.2326 | 0.3114 | 0.3521 | 0.4861 | 0.4861 | 0.5751 | 0.2984 | 0.3552 | 0.3559 | 0.0236* | 0.0398* | 0.0534* | 0.0542* | 0.4100 | 0.2479 | 0.2574 | 0.2574 | 0.0579* | 0.0837 | 0.0523* | 0.0698 |

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
| 32 | aud_kzs_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_kzs_L4 | 1.00 | 1.00 | 1.00 |
| 32 | aud_kzo_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_kzo_L4 | 1.00 | 1.00 | 1.00 |
| 32 | aud_kcs_L1 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kcs_L4 | 1.00 | 1.00 | 1.00 |
| 8 | reflex | 1.00 | 1.00 | 1.00 |
| 8 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 8 | aud_L1 | 1.00 | 1.00 | 0.00 |
| 8 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 8 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzo_L1 | 1.00 | 1.00 | 0.00 |
| 8 | aud_kzo_L4 | 1.00 | 1.00 | 1.00 |
| 8 | aud_kcs_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kcs_L4 | 0.00 | 0.00 | 0.00 |
| 2 | reflex | 0.06 | 0.00 | 0.00 |
| 2 | reflex_ec0 | 1.00 | 1.00 | 0.00 |
| 2 | aud_L1 | 1.00 | 0.00 | 0.00 |
| 2 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 2 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzo_L1 | 1.00 | 1.00 | 1.00 |
| 2 | aud_kzo_L4 | 1.00 | 0.00 | 0.00 |
| 2 | aud_kcs_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kcs_L4 | 0.00 | 0.00 | 0.00 |

## The sweep, e_listen (the listener's clock)

### `aud` — plant audition at the seam

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | s2_L1 | s2_L2 | s2_L3 | s2_L4 | kzs_L1 | kzs_L2 | kzs_L3 | kzs_L4 | kzo_L1 | kzo_L2 | kzo_L3 | kzo_L4 | kcs_L1 | kcs_L2 | kcs_L3 | kcs_L4 | kco_L1 | kco_L2 | kco_L3 | kco_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0046 | 0.0435* | 0.0201* | 0.0247* | 0.0260* | 0.0435* | 0.0201* | 0.0247* | 0.0260* | 0.0435* | 0.0201* | 0.0247* | 0.0260* | 0.0416* | 0.0361* | 0.0308* | 0.0214* | 0.0416* | 0.0361* | 0.0308* | 0.0214* | 0.0329* | 0.0252* | 0.0271* | 0.0287* | 0.0329* | 0.0252* | 0.0271* | 0.0287* |
| 8 | 0.0611 | 0.0118 | 0.0300* | 0.0243* | 0.0077* | 0.0096* | 0.0966 | 0.1241 | 0.1533 | 0.1403 | 0.2806 | 0.2488 | 0.3180 | 0.3903 | 0.1257 | 0.2016 | 0.2819 | 0.2657 | 0.0305* | 0.0216* | 0.0177* | 0.0206* | 0.0801 | 0.1240 | 0.2621 | 0.2583 | 0.0313* | 0.0129* | 0.0144* | 0.0227* |
| 2 | 0.0611 | 0.0472 | 0.0359* | 0.0211* | 0.0308* | 0.0061* | 0.1509 | 0.1707 | 0.1802 | 0.1884 | 0.3873 | 0.3088 | 0.2972 | 0.3773 | 0.2297 | 0.2095 | 0.1769 | 0.2841 | 0.0193* | 0.0315* | 0.0490* | 0.0566* | 0.3226 | 0.1898 | 0.1588 | 0.2533 | 0.0381* | 0.0499* | 0.0795 | 0.0677 |

### `key` — frozen posture key, no audition

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | s2_L1 | s2_L2 | s2_L3 | s2_L4 | kzs_L1 | kzs_L2 | kzs_L3 | kzs_L4 | kzo_L1 | kzo_L2 | kzo_L3 | kzo_L4 | kcs_L1 | kcs_L2 | kcs_L3 | kcs_L4 | kco_L1 | kco_L2 | kco_L3 | kco_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0046 | 0.0434* | 0.0754 | 0.0205* | 0.0205* | 0.0434* | 0.0754 | 0.0205* | 0.0205* | 0.0434* | 0.0754 | 0.0205* | 0.0205* | 0.0361* | 0.0789 | 0.0754 | 0.0787 | 0.0361* | 0.0789 | 0.0754 | 0.0787 | 0.0462* | 0.0590* | 0.0361* | 0.0397* | 0.0462* | 0.0590* | 0.0361* | 0.0397* |
| 8 | 0.0611 | 0.0118 | 0.0521* | 0.0469* | 0.0331* | 0.0191* | 0.1280 | 0.2193 | 0.1855 | 0.1855 | 0.9190 | 0.4658 | 0.7791 | 0.9157 | 0.5138 | 0.8405 | 0.4306 | 0.4271 | 0.0287* | 0.0281* | 0.0419* | 0.0319* | 0.6193 | 0.8971 | 0.4131 | 0.4738 | 0.0277* | 0.0076* | 0.0387* | 0.0270* |
| 2 | 0.0611 | 0.0472 | 0.0339* | 0.0177* | 0.0094* | 0.0051* | 0.1939 | 0.1708 | 0.1794 | 0.1794 | 0.2972 | 0.2756 | 0.4283 | 0.4283 | 0.5011 | 0.2680 | 0.3344 | 0.3350 | 0.0209* | 0.0396* | 0.0529* | 0.0537* | 0.3630 | 0.2189 | 0.2273 | 0.2273 | 0.0487* | 0.0726 | 0.0447* | 0.0616 |

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
| 32 | aud_kzs_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_kzs_L4 | 1.00 | 1.00 | 1.00 |
| 32 | aud_kzo_L1 | 1.00 | 0.00 | 0.00 |
| 32 | aud_kzo_L4 | 1.00 | 1.00 | 1.00 |
| 32 | aud_kcs_L1 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kcs_L4 | 1.00 | 1.00 | 1.00 |
| 8 | reflex | 1.00 | 1.00 | 1.00 |
| 8 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 8 | aud_L1 | 1.00 | 1.00 | 1.00 |
| 8 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 8 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzo_L1 | 1.00 | 1.00 | 1.00 |
| 8 | aud_kzo_L4 | 1.00 | 1.00 | 1.00 |
| 8 | aud_kcs_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kcs_L4 | 0.00 | 0.00 | 0.00 |
| 2 | reflex | 1.00 | 0.06 | 0.00 |
| 2 | reflex_ec0 | 1.00 | 1.00 | 1.00 |
| 2 | aud_L1 | 1.00 | 1.00 | 0.00 |
| 2 | aud_L4 | 1.00 | 1.00 | 1.00 |
| 2 | aud_s0_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_s0_L4 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzo_L1 | 1.00 | 1.00 | 1.00 |
| 2 | aud_kzo_L4 | 1.00 | 0.00 | 0.00 |
| 2 | aud_kcs_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kcs_L4 | 0.00 | 0.00 | 0.00 |

## The transfer question, isolated

Three numbers per (tempo, level), all `aud`, all in `e_piece`: the ORACLE content harvested at the asked tempo (`rec` — not deployable), the same-tempo kinematic conversion (`kzo`, s = 1 — the CONVERSION COST alone), and the kinematic unit carried from the slowest tempo (`kzs` — the transfer). `kzs / kzo` is what the tempo change costs once the conversion is paid for; `kzo / rec` is what the conversion costs.

| H | L | band | rec | kzo | kzs | scl0 | kzo/rec | kzs/kzo | kzs/scl0 | sat(kzs) |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.0611 | 0.0454* | 0.0408* | 0.0408* | 0.0454* | 0.90x | 1.00x | 0.90x | 0.000 |
| 32 | 2 | 0.0611 | 0.0266* | 0.0399* | 0.0399* | 0.0266* | 1.50x | 1.00x | 1.50x | 0.000 |
| 32 | 3 | 0.0611 | 0.0259* | 0.0326* | 0.0326* | 0.0259* | 1.26x | 1.00x | 1.26x | 0.000 |
| 32 | 4 | 0.0611 | 0.0263* | 0.0213* | 0.0213* | 0.0263* | 0.81x | 1.00x | 0.81x | 0.000 |
| 8 | 1 | 0.0611 | 0.0312* | 0.0312* | 0.1611 | 0.1093 | 1.00x | 5.17x | 1.47x | 0.000 |
| 8 | 2 | 0.0611 | 0.0289* | 0.0243* | 0.2340 | 0.1412 | 0.84x | 9.64x | 1.66x | 0.000 |
| 8 | 3 | 0.0611 | 0.0075* | 0.0183* | 0.3144 | 0.1742 | 2.43x | 17.20x | 1.80x | 0.000 |
| 8 | 4 | 0.0611 | 0.0123* | 0.0226* | 0.2944 | 0.1461 | 1.84x | 13.04x | 2.01x | 0.000 |
| 2 | 1 | 0.0611 | 0.0453* | 0.0222* | 0.3999 | 0.2168 | 0.49x | 18.04x | 1.84x | 0.000 |
| 2 | 2 | 0.0611 | 0.0298* | 0.0343* | 0.2626 | 0.2323 | 1.15x | 7.65x | 1.13x | 0.000 |
| 2 | 3 | 0.0611 | 0.0334* | 0.0501* | 0.1968 | 0.2356 | 1.50x | 3.93x | 0.84x | 0.000 |
| 2 | 4 | 0.0611 | 0.0069* | 0.0578* | 0.3184 | 0.2452 | 8.38x | 5.51x | 1.30x | 0.000 |

## The derivative scheme, measured

The exact (`zoh`) inverse against the continuous-time (`ct`) one the brief names, at the same content and the same source. They differ only in the coefficient on acceleration, by 1.4528x.

| H | L | kzo | kco | kco/kzo | kzs | kcs | kcs/kzs |
|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.0408 | 0.0313 | 0.77x | 0.0408 | 0.0313 | 0.77x |
| 32 | 2 | 0.0399 | 0.0272 | 0.68x | 0.0399 | 0.0272 | 0.68x |
| 32 | 3 | 0.0326 | 0.0249 | 0.76x | 0.0326 | 0.0249 | 0.76x |
| 32 | 4 | 0.0213 | 0.0298 | 1.40x | 0.0213 | 0.0298 | 1.40x |
| 8 | 1 | 0.0312 | 0.0333 | 1.07x | 0.1611 | 0.1059 | 0.66x |
| 8 | 2 | 0.0243 | 0.0143 | 0.59x | 0.2340 | 0.1525 | 0.65x |
| 8 | 3 | 0.0183 | 0.0164 | 0.90x | 0.3144 | 0.2958 | 0.94x |
| 8 | 4 | 0.0226 | 0.0261 | 1.16x | 0.2944 | 0.2873 | 0.98x |
| 2 | 1 | 0.0222 | 0.0508 | 2.29x | 0.3999 | 0.4369 | 1.09x |
| 2 | 2 | 0.0343 | 0.0674 | 1.96x | 0.2626 | 0.2827 | 1.08x |
| 2 | 3 | 0.0501 | 0.0907 | 1.81x | 0.1968 | 0.2311 | 1.17x |
| 2 | 4 | 0.0578 | 0.0750 | 1.30x | 0.3184 | 0.2850 | 0.90x |

## Saturation: the actuator ceiling

Fraction of scalar command components the clip bound when the path was converted, per (arm, tempo, level), non-poison members only.

| arm | H32 L1 | H32 L2 | H32 L3 | H32 L4 | H8 L1 | H8 L2 | H8 L3 | H8 L4 | H2 L1 | H2 L2 | H2 L3 | H2 L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kzs | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kzo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kcs | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kco | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## The era table: which content wins at each tempo

Best arm at each tempo under each grade, over every committed arm (the incumbent and its two instruments excluded), and the same over the DEPLOYABLE arms only — those whose content exists without practising at the asked tempo (`s0`, `s2`, `kzs`, `kcs`).

| H | best (any) | e | in band | best DEPLOYABLE | e | in band | reflex | ratio |
|---|---|---|---|---|---|---|---|---|
| 32 | key_L3 | 0.0175 | yes | key_s0_L3 | 0.0175 | yes | 0.0037 | 0.21x |
| 8 | aud_L3 | 0.0075 | yes | aud_kcs_L1 | 0.1059 | no | 0.0174 | 0.16x |
| 2 | aud_L4 | 0.0069 | yes | aud_kzs_L3 | 0.1968 | no | 0.0690 | 0.35x |

## The ledger, drilled only (per performer)

| H | arm | n_fb | n_ground | t_piece (s) |
|---|---|---|---|---|
| 32 | reflex | 256.0 | 0.0 | 31.744 |
| 32 | reflex_ec | 256.0 | 0.0 | 31.744 |
| 32 | aud_L4 | 1.0 | 24.0 | 156.100 |
| 32 | key_L4 | 1.0 | 0.0 | 6.244 |
| 32 | aud_kzs_L4 | 1.0 | 24.0 | 156.100 |
| 32 | key_kzs_L4 | 1.0 | 0.0 | 6.244 |
| 32 | aud_kzs_L1 | 8.0 | 192.0 | 173.600 |
| 32 | key_kzs_L1 | 8.0 | 0.0 | 6.944 |
| 8 | reflex | 64.0 | 0.0 | 7.936 |
| 8 | reflex_ec | 64.0 | 0.0 | 7.936 |
| 8 | aud_L4 | 1.0 | 24.0 | 40.900 |
| 8 | key_L4 | 1.0 | 0.0 | 1.636 |
| 8 | aud_kzs_L4 | 1.0 | 24.0 | 40.900 |
| 8 | key_kzs_L4 | 1.0 | 0.0 | 1.636 |
| 8 | aud_kzs_L1 | 8.0 | 192.0 | 58.400 |
| 8 | key_kzs_L1 | 8.0 | 0.0 | 2.336 |
| 2 | reflex | 16.0 | 0.0 | 1.984 |
| 2 | reflex_ec | 16.0 | 0.0 | 1.984 |
| 2 | aud_L4 | 1.0 | 24.0 | 12.100 |
| 2 | key_L4 | 1.0 | 0.0 | 0.484 |
| 2 | aud_kzs_L4 | 1.0 | 24.0 | 12.100 |
| 2 | key_kzs_L4 | 1.0 | 0.0 | 0.484 |
| 2 | aud_kzs_L1 | 8.0 | 192.0 | 29.600 |
| 2 | key_kzs_L1 | 8.0 | 0.0 | 1.184 |

