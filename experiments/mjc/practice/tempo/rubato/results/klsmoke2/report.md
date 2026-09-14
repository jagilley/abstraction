# rubato / klsmoke2 — Phase 1: the ceiling of the factored program

seed 0 · R* 0.16 · tempi [32, 16, 8, 2] · Delta 5 (120 ms) · app=fixed · 1516s (library 606 · kin 2 · sweep 667)
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

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | kzs_L1 | kzs_L2 | kzs_L3 | kzs_L4 | kzoa_L1 | kzoa_L2 | kzoa_L3 | kzoa_L4 | kzsa_L1 | kzsa_L2 | kzsa_L3 | kzsa_L4 | kfl_L1 | kfl_L2 | kfl_L3 | kfl_L4 | kfm_L1 | kfm_L2 | kfm_L3 | kfm_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0037 | 0.0454* | 0.0266* | 0.0259* | 0.0263* | 0.0454* | 0.0266* | 0.0259* | 0.0263* | 0.0408* | 0.0399* | 0.0326* | 0.0213* | 0.0408* | 0.0399* | 0.0326* | 0.0213* | 0.0408* | 0.0399* | 0.0326* | 0.0213* | 0.0260* | 0.0297* | 0.0446* | 0.0429* | 0.0405* | 0.0823 | 0.0490* | 0.0811 |
| 16 | 0.0611 | 0.0103 | 0.0374* | 0.0401* | 0.0465* | 0.0093* | 0.0677 | 0.0623 | 0.0858 | 0.0951 | 0.1227 | 0.1321 | 0.1478 | 0.2940 | 0.0404* | 0.0253* | 0.0416* | 0.0271* | 0.0836 | 0.0426* | 0.0426* | 0.0446* | 0.0913 | 0.0460* | 0.1183 | 0.0795 | 0.0781 | 0.0800 | 0.0676 | 0.0999 |
| 8 | 0.0611 | 0.0174 | 0.0312* | 0.0289* | 0.0075* | 0.0123* | 0.1093 | 0.1412 | 0.1742 | 0.1461 | 0.1611 | 0.2340 | 0.3144 | 0.2944 | 0.0312* | 0.0243* | 0.0183* | 0.0226* | 0.0868 | 0.0838 | 0.0529* | 0.0550* | 0.0965 | 0.0520* | 0.0555* | 0.0663 | 0.0690 | 0.0593* | 0.0575* | 0.0848 |
| 2 | 0.0611 | 0.0690 | 0.0453* | 0.0298* | 0.0334* | 0.0069* | 0.2168 | 0.2323 | 0.2356 | 0.2452 | 0.3999 | 0.2626 | 0.1968 | 0.3184 | 0.0222* | 0.0343* | 0.0501* | 0.0578* | 0.3498 | 0.2033 | 0.1210 | 0.0660 | 0.4119 | 0.1735 | 0.1163 | 0.0893 | 0.2788 | 0.1494 | 0.1805 | 0.2210 |

### `key` — frozen posture key, no audition

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | kzs_L1 | kzs_L2 | kzs_L3 | kzs_L4 | kzoa_L1 | kzoa_L2 | kzoa_L3 | kzoa_L4 | kzsa_L1 | kzsa_L2 | kzsa_L3 | kzsa_L4 | kfl_L1 | kfl_L2 | kfl_L3 | kfl_L4 | kfm_L1 | kfm_L2 | kfm_L3 | kfm_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0037 | 0.0562* | 0.0842 | 0.0175* | 0.0175* | 0.0562* | 0.0842 | 0.0175* | 0.0175* | 0.0399* | 0.0872 | 0.0803 | 0.0839 | 0.0399* | 0.0872 | 0.0803 | 0.0839 | 0.0399* | 0.0872 | 0.0803 | 0.0839 | 0.0522* | 0.0997 | 0.0604* | 0.0627 | 0.0689 | 0.1587 | 0.0850 | 0.0862 |
| 16 | 0.0611 | 0.0103 | 0.0549* | 0.0188* | 0.0142* | 0.0087* | 0.0962 | 0.1087 | 0.0990 | 0.0990 | 0.8547 | 0.6246 | 0.4335 | 0.6022 | 0.0395* | 0.0370* | 0.0487* | 0.0287* | 0.0776 | 0.0339* | 0.0498* | 0.0719 | 0.1202 | 0.0694 | 0.0673 | 0.0821 | 0.0733 | 0.0662 | 0.0631 | 0.0829 |
| 8 | 0.0611 | 0.0174 | 0.0580* | 0.0544* | 0.0310* | 0.0185* | 0.1400 | 0.2349 | 0.2015 | 0.2015 | 0.5547 | 0.8980 | 0.4709 | 0.4651 | 0.0326* | 0.0288* | 0.0429* | 0.0348* | 0.1285 | 0.0545* | 0.0587* | 0.0498* | 0.1420 | 0.0564* | 0.0555* | 0.0574* | 0.1273 | 0.0665 | 0.0619 | 0.0608* |
| 2 | 0.0611 | 0.0690 | 0.0399* | 0.0181* | 0.0123* | 0.0074* | 0.2403 | 0.2234 | 0.2326 | 0.2326 | 0.5751 | 0.2984 | 0.3552 | 0.3559 | 0.0236* | 0.0398* | 0.0534* | 0.0542* | 0.2919 | 0.0846 | 0.1167 | 0.0938 | 0.2950 | 0.1260 | 0.1398 | 0.1160 | 0.3088 | 0.2225 | 0.1686 | 0.1430 |

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
| 32 | aud_kzs_L2 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kzs_L3 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kzs_L4 | 1.00 | 1.00 | 1.00 |
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
| 16 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzs_L2 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzs_L3 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
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
| 8 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L2 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L3 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
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
| 2 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L2 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L3 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L2 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L3 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L4 | 0.00 | 0.00 | 0.00 |

## The sweep, e_listen (the listener's clock)

### `aud` — plant audition at the seam

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | kzs_L1 | kzs_L2 | kzs_L3 | kzs_L4 | kzoa_L1 | kzoa_L2 | kzoa_L3 | kzoa_L4 | kzsa_L1 | kzsa_L2 | kzsa_L3 | kzsa_L4 | kfl_L1 | kfl_L2 | kfl_L3 | kfl_L4 | kfm_L1 | kfm_L2 | kfm_L3 | kfm_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0046 | 0.0435* | 0.0201* | 0.0247* | 0.0260* | 0.0435* | 0.0201* | 0.0247* | 0.0260* | 0.0416* | 0.0361* | 0.0308* | 0.0214* | 0.0416* | 0.0361* | 0.0308* | 0.0214* | 0.0416* | 0.0361* | 0.0308* | 0.0214* | 0.0292* | 0.0317* | 0.0387* | 0.0416* | 0.0419* | 0.0681 | 0.0415* | 0.0777 |
| 16 | 0.0611 | 0.0082 | 0.0330* | 0.0386* | 0.0433* | 0.0071* | 0.0599* | 0.0557* | 0.0792 | 0.0926 | 0.1218 | 0.1219 | 0.1307 | 0.2487 | 0.0390* | 0.0289* | 0.0417* | 0.0286* | 0.0784 | 0.0408* | 0.0422* | 0.0441* | 0.0868 | 0.0414* | 0.1038 | 0.0719 | 0.0771 | 0.0742 | 0.0595* | 0.0908 |
| 8 | 0.0611 | 0.0118 | 0.0300* | 0.0243* | 0.0077* | 0.0096* | 0.0966 | 0.1241 | 0.1533 | 0.1403 | 0.1257 | 0.2016 | 0.2819 | 0.2657 | 0.0305* | 0.0216* | 0.0177* | 0.0206* | 0.0785 | 0.0774 | 0.0485* | 0.0485* | 0.0870 | 0.0394* | 0.0476* | 0.0651 | 0.0604* | 0.0553* | 0.0538* | 0.0825 |
| 2 | 0.0611 | 0.0472 | 0.0359* | 0.0211* | 0.0308* | 0.0061* | 0.1509 | 0.1707 | 0.1802 | 0.1884 | 0.2297 | 0.2095 | 0.1769 | 0.2841 | 0.0193* | 0.0315* | 0.0490* | 0.0566* | 0.2553 | 0.1666 | 0.1141 | 0.0568* | 0.3036 | 0.1369 | 0.0986 | 0.0728 | 0.1392 | 0.1218 | 0.1697 | 0.2022 |

### `key` — frozen posture key, no audition

| H | band | reflex | rec_L1 | rec_L2 | rec_L3 | rec_L4 | s0_L1 | s0_L2 | s0_L3 | s0_L4 | kzs_L1 | kzs_L2 | kzs_L3 | kzs_L4 | kzoa_L1 | kzoa_L2 | kzoa_L3 | kzoa_L4 | kzsa_L1 | kzsa_L2 | kzsa_L3 | kzsa_L4 | kfl_L1 | kfl_L2 | kfl_L3 | kfl_L4 | kfm_L1 | kfm_L2 | kfm_L3 | kfm_L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0611 | 0.0046 | 0.0434* | 0.0754 | 0.0205* | 0.0205* | 0.0434* | 0.0754 | 0.0205* | 0.0205* | 0.0361* | 0.0789 | 0.0754 | 0.0787 | 0.0361* | 0.0789 | 0.0754 | 0.0787 | 0.0361* | 0.0789 | 0.0754 | 0.0787 | 0.0483* | 0.0877 | 0.0542* | 0.0562* | 0.0618 | 0.1457 | 0.0796 | 0.0807 |
| 16 | 0.0611 | 0.0082 | 0.0450* | 0.0193* | 0.0151* | 0.0090* | 0.0849 | 0.0984 | 0.0924 | 0.0924 | 0.7767 | 0.5899 | 0.4104 | 0.5643 | 0.0324* | 0.0331* | 0.0467* | 0.0292* | 0.0753 | 0.0305* | 0.0425* | 0.0628 | 0.1169 | 0.0609* | 0.0577* | 0.0721 | 0.0741 | 0.0609* | 0.0559* | 0.0748 |
| 8 | 0.0611 | 0.0118 | 0.0521* | 0.0469* | 0.0331* | 0.0191* | 0.1280 | 0.2193 | 0.1855 | 0.1855 | 0.5138 | 0.8405 | 0.4306 | 0.4271 | 0.0287* | 0.0281* | 0.0419* | 0.0319* | 0.1203 | 0.0460* | 0.0541* | 0.0450* | 0.1348 | 0.0505* | 0.0529* | 0.0526* | 0.1204 | 0.0632 | 0.0592* | 0.0568* |
| 2 | 0.0611 | 0.0472 | 0.0339* | 0.0177* | 0.0094* | 0.0051* | 0.1939 | 0.1708 | 0.1794 | 0.1794 | 0.5011 | 0.2680 | 0.3344 | 0.3350 | 0.0209* | 0.0396* | 0.0529* | 0.0537* | 0.2518 | 0.0796 | 0.1095 | 0.0847 | 0.2493 | 0.1146 | 0.1292 | 0.1013 | 0.2673 | 0.2115 | 0.1580 | 0.1326 |

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
| 32 | aud_kzs_L2 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kzs_L3 | 1.00 | 1.00 | 0.00 |
| 32 | aud_kzs_L4 | 1.00 | 1.00 | 1.00 |
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
| 16 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzs_L2 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzs_L3 | 0.00 | 0.00 | 0.00 |
| 16 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
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
| 8 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L2 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L3 | 0.00 | 0.00 | 0.00 |
| 8 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
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
| 2 | aud_kzs_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L2 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L3 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzs_L4 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L1 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L2 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L3 | 0.00 | 0.00 | 0.00 |
| 2 | aud_kzsa_L4 | 1.00 | 0.00 | 0.00 |

## The transfer question, isolated

Three numbers per (tempo, level), all `aud`, all in `e_piece`: the ORACLE content harvested at the asked tempo (`rec` — not deployable), the same-tempo kinematic conversion (`kzo`, s = 1 — the CONVERSION COST alone), and the kinematic unit carried from the slowest tempo (`kzs` — the transfer). `kzs / kzo` is what the tempo change costs once the conversion is paid for; `kzo / rec` is what the conversion costs.

| H | L | band | rec | kzo | kzs | scl0 | kzo/rec | kzs/kzo | kzs/scl0 | sat(kzs) |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 0.0611 | 0.0454* | — | 0.0408* | 0.0454* | — | — | 0.90x | 0.000 |
| 32 | 2 | 0.0611 | 0.0266* | — | 0.0399* | 0.0266* | — | — | 1.50x | 0.000 |
| 32 | 3 | 0.0611 | 0.0259* | — | 0.0326* | 0.0259* | — | — | 1.26x | 0.000 |
| 32 | 4 | 0.0611 | 0.0263* | — | 0.0213* | 0.0263* | — | — | 0.81x | 0.000 |
| 16 | 1 | 0.0611 | 0.0374* | — | 0.1227 | 0.0677 | — | — | 1.81x | 0.000 |
| 16 | 2 | 0.0611 | 0.0401* | — | 0.1321 | 0.0623 | — | — | 2.12x | 0.000 |
| 16 | 3 | 0.0611 | 0.0465* | — | 0.1478 | 0.0858 | — | — | 1.72x | 0.000 |
| 16 | 4 | 0.0611 | 0.0093* | — | 0.2940 | 0.0951 | — | — | 3.09x | 0.000 |
| 8 | 1 | 0.0611 | 0.0312* | — | 0.1611 | 0.1093 | — | — | 1.47x | 0.000 |
| 8 | 2 | 0.0611 | 0.0289* | — | 0.2340 | 0.1412 | — | — | 1.66x | 0.000 |
| 8 | 3 | 0.0611 | 0.0075* | — | 0.3144 | 0.1742 | — | — | 1.80x | 0.000 |
| 8 | 4 | 0.0611 | 0.0123* | — | 0.2944 | 0.1461 | — | — | 2.01x | 0.000 |
| 2 | 1 | 0.0611 | 0.0453* | — | 0.3999 | 0.2168 | — | — | 1.84x | 0.000 |
| 2 | 2 | 0.0611 | 0.0298* | — | 0.2626 | 0.2323 | — | — | 1.13x | 0.000 |
| 2 | 3 | 0.0611 | 0.0334* | — | 0.1968 | 0.2356 | — | — | 0.84x | 0.000 |
| 2 | 4 | 0.0611 | 0.0069* | — | 0.3184 | 0.2452 | — | — | 1.30x | 0.000 |

## The derivative scheme, measured

The exact (`zoh`) inverse against the continuous-time (`ct`) one the brief names, at the same content and the same source. They differ only in the coefficient on acceleration, by 1.4528x.

| H | L | kzo | kco | kco/kzo | kzs | kcs | kcs/kzs |
|---|---|---|---|---|---|---|---|
| 32 | 1 | — | — | — | 0.0408 | — | — |
| 32 | 2 | — | — | — | 0.0399 | — | — |
| 32 | 3 | — | — | — | 0.0326 | — | — |
| 32 | 4 | — | — | — | 0.0213 | — | — |
| 16 | 1 | — | — | — | 0.1227 | — | — |
| 16 | 2 | — | — | — | 0.1321 | — | — |
| 16 | 3 | — | — | — | 0.1478 | — | — |
| 16 | 4 | — | — | — | 0.2940 | — | — |
| 8 | 1 | — | — | — | 0.1611 | — | — |
| 8 | 2 | — | — | — | 0.2340 | — | — |
| 8 | 3 | — | — | — | 0.3144 | — | — |
| 8 | 4 | — | — | — | 0.2944 | — | — |
| 2 | 1 | — | — | — | 0.3999 | — | — |
| 2 | 2 | — | — | — | 0.2626 | — | — |
| 2 | 3 | — | — | — | 0.1968 | — | — |
| 2 | 4 | — | — | — | 0.3184 | — | — |

## Saturation: the actuator ceiling

Fraction of scalar command components the clip bound when the path was converted, per (arm, tempo, level), non-poison members only.

| arm | H32 L1 | H32 L2 | H32 L3 | H32 L4 | H16 L1 | H16 L2 | H16 L3 | H16 L4 | H8 L1 | H8 L2 | H8 L3 | H8 L4 | H2 L1 | H2 L2 | H2 L3 | H2 L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kfl | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kfm | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kzoa | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kzs | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| kzsa | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## The era table: which content wins at each tempo

Best arm at each tempo under each grade, over every committed arm (the incumbent and its two instruments excluded), and the same over the DEPLOYABLE arms only — those whose content exists without practising at the asked tempo (`s0`, `s2`, `kzs`, `kcs`).

| H | best (any) | e | in band | best DEPLOYABLE | e | in band | reflex | ratio |
|---|---|---|---|---|---|---|---|---|
| 32 | key_L3 | 0.0175 | yes | key_s0_L3 | 0.0175 | yes | 0.0037 | 0.21x |
| 16 | key_L4 | 0.0087 | yes | key_kzsa_L2 | 0.0339 | yes | 0.0103 | 0.30x |
| 8 | aud_L3 | 0.0075 | yes | key_kzsa_L4 | 0.0498 | yes | 0.0174 | 0.35x |
| 2 | aud_L4 | 0.0069 | yes | aud_kzsa_L4 | 0.0660 | no | 0.0690 | 1.05x |

## Parity per slot against the same-tempo recording

A slot is OPEN at a tempo iff the arm's content, EXECUTED ON THE PLANT from that slot's HELD-OUT launch states, is at least as good as the same-tempo recording's own audition winner on at least tau = 0.75 of them. `solo`'s gate with the tempo added to the index; the reference is the per-state winner, not a fixed member. The reference side is an instrument and is not charged.

| arm | H32 L1 | H32 L2 | H32 L3 | H32 L4 | H16 L1 | H16 L2 | H16 L3 | H16 L4 | H8 L1 | H8 L2 | H8 L3 | H8 L4 | H2 L1 | H2 L2 | H2 L3 | H2 L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kfl | 18/48 | 7/24 | 0/12 | 0/6 | 8/48 | 1/24 | 0/12 | 0/6 | 19/48 | 2/24 | 0/12 | 0/6 | 12/48 | 4/24 | 1/12 | 0/6 |
| kfm | 11/48 | 4/24 | 0/12 | 0/6 | 11/48 | 2/24 | 0/12 | 0/6 | 24/48 | 1/24 | 0/12 | 0/6 | 9/48 | 3/24 | 0/12 | 0/6 |
| kzoa | 28/48 | 15/24 | 6/12 | 0/6 | 28/48 | 8/24 | 4/12 | 0/6 | 28/48 | 8/24 | 6/12 | 0/6 | 14/48 | 7/24 | 0/12 | 0/6 |
| kzs | 28/48 | 15/24 | 6/12 | 0/6 | 6/48 | 0/24 | 0/12 | 0/6 | 6/48 | 2/24 | 0/12 | 0/6 | 8/48 | 5/24 | 0/12 | 0/6 |
| kzsa | 28/48 | 15/24 | 6/12 | 0/6 | 19/48 | 5/24 | 0/12 | 0/6 | 22/48 | 4/24 | 0/12 | 0/6 | 13/48 | 4/24 | 1/12 | 0/6 |

Mean executed error on the held-out states, arm vs the recording it is measured against (arm / reference):

| arm | H32 L1 | H32 L2 | H32 L3 | H32 L4 | H16 L1 | H16 L2 | H16 L3 | H16 L4 | H8 L1 | H8 L2 | H8 L3 | H8 L4 | H2 L1 | H2 L2 | H2 L3 | H2 L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kfl | 0.0382/0.0350 | 0.0559/0.0382 | 0.1045/0.0589 | 0.0523/0.0194 | 0.0637/0.0429 | 0.0791/0.0427 | 0.1320/0.0436 | 0.0822/0.0092 | 0.0522/0.0448 | 0.0705/0.0444 | 0.0835/0.0336 | 0.0534/0.0065 | 0.0563/0.0352 | 0.0778/0.0369 | 0.0714/0.0207 | 0.0669/0.0062 |
| kfm | 0.0451/0.0350 | 0.0567/0.0382 | 0.1028/0.0589 | 0.0754/0.0194 | 0.0564/0.0429 | 0.0706/0.0427 | 0.1160/0.0436 | 0.0774/0.0092 | 0.0491/0.0448 | 0.0657/0.0444 | 0.0770/0.0336 | 0.0483/0.0065 | 0.0647/0.0352 | 0.0895/0.0369 | 0.0745/0.0207 | 0.0738/0.0062 |
| kzoa | 0.0365/0.0350 | 0.0373/0.0382 | 0.0625/0.0589 | 0.0481/0.0194 | 0.0412/0.0429 | 0.0465/0.0427 | 0.0486/0.0436 | 0.0369/0.0092 | 0.0476/0.0448 | 0.0466/0.0444 | 0.0360/0.0336 | 0.0253/0.0065 | 0.0392/0.0352 | 0.0387/0.0369 | 0.0239/0.0207 | 0.0389/0.0062 |
| kzs | 0.0365/0.0350 | 0.0373/0.0382 | 0.0625/0.0589 | 0.0481/0.0194 | 0.1288/0.0429 | 0.1910/0.0427 | 0.3182/0.0436 | 0.4835/0.0092 | 0.1239/0.0448 | 0.1863/0.0444 | 0.3204/0.0336 | 0.5183/0.0065 | 0.0815/0.0352 | 0.1137/0.0369 | 0.1805/0.0207 | 0.2822/0.0062 |
| kzsa | 0.0365/0.0350 | 0.0373/0.0382 | 0.0625/0.0589 | 0.0481/0.0194 | 0.0477/0.0429 | 0.0603/0.0427 | 0.1006/0.0436 | 0.0539/0.0092 | 0.0450/0.0448 | 0.0593/0.0444 | 0.0599/0.0336 | 0.0408/0.0065 | 0.0516/0.0352 | 0.0706/0.0369 | 0.0573/0.0207 | 0.0577/0.0062 |

## Phase 2 — the learned cerebellum

**The linear family's recovered coefficients**, against the analytic inverse it is trying to be. Fitted as a general 2x4 map with a bias, so the isotropy and the which-channel-depends-on-what were NOT given to it.

| | fitted (x, y) | analytic |
|---|---|---|
| drag `c_v` | 0.168617, 0.159768 | 0.200000 |
| accel `c_a` | 0.008360, 0.008296 | 0.008717 |
| off-diagonal (max abs) | 1.84e-02 | 0 |
| bias (max abs) | 3.58e-03 | 0 |

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
| 32 | 1 | 0.0611 | 0.0408* | 0.0260* | 0.0405* | 0.64x | 0.99x |
| 32 | 2 | 0.0611 | 0.0399* | 0.0297* | 0.0823 | 0.74x | 2.06x |
| 32 | 3 | 0.0611 | 0.0326* | 0.0446* | 0.0490* | 1.37x | 1.50x |
| 32 | 4 | 0.0611 | 0.0213* | 0.0429* | 0.0811 | 2.01x | 3.80x |
| 16 | 1 | 0.0611 | 0.0836 | 0.0913 | 0.0781 | 1.09x | 0.93x |
| 16 | 2 | 0.0611 | 0.0426* | 0.0460* | 0.0800 | 1.08x | 1.88x |
| 16 | 3 | 0.0611 | 0.0426* | 0.1183 | 0.0676 | 2.78x | 1.59x |
| 16 | 4 | 0.0611 | 0.0446* | 0.0795 | 0.0999 | 1.78x | 2.24x |
| 8 | 1 | 0.0611 | 0.0868 | 0.0965 | 0.0690 | 1.11x | 0.79x |
| 8 | 2 | 0.0611 | 0.0838 | 0.0520* | 0.0593* | 0.62x | 0.71x |
| 8 | 3 | 0.0611 | 0.0529* | 0.0555* | 0.0575* | 1.05x | 1.09x |
| 8 | 4 | 0.0611 | 0.0550* | 0.0663 | 0.0848 | 1.20x | 1.54x |
| 2 | 1 | 0.0611 | 0.3498 | 0.4119 | 0.2788 | 1.18x | 0.80x |
| 2 | 2 | 0.0611 | 0.2033 | 0.1735 | 0.1494 | 0.85x | 0.73x |
| 2 | 3 | 0.0611 | 0.1210 | 0.1163 | 0.1805 | 0.96x | 1.49x |
| 2 | 4 | 0.0611 | 0.0660 | 0.0893 | 0.2210 | 1.35x | 3.35x |

**In-band cells: ceiling 9, linear 7, MLP 4** (of 16).

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
| 32 | aud_kzs_L4 | 1.0 | 24.0 | 156.100 |
| 32 | key_kzs_L4 | 1.0 | 0.0 | 6.244 |
| 32 | aud_kzs_L1 | 8.0 | 192.0 | 173.600 |
| 32 | key_kzs_L1 | 8.0 | 0.0 | 6.944 |
| 16 | reflex | 128.0 | 0.0 | 15.872 |
| 16 | reflex_ec | 128.0 | 0.0 | 15.872 |
| 16 | aud_L4 | 1.0 | 24.0 | 79.300 |
| 16 | key_L4 | 1.0 | 0.0 | 3.172 |
| 16 | aud_kzs_L4 | 1.0 | 24.0 | 79.300 |
| 16 | key_kzs_L4 | 1.0 | 0.0 | 3.172 |
| 16 | aud_kzs_L1 | 8.0 | 192.0 | 96.800 |
| 16 | key_kzs_L1 | 8.0 | 0.0 | 3.872 |
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

