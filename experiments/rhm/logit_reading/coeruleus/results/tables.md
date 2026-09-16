# coeruleus -- reduced tables

## Q1. The prize after a corrupted token, and what a one-knob consumer reaches

### eps-trained 64k (kappa*_eps = 4.85, 1577 isolated events, H = 8)

| tau | H(q) | H(p_L^eps) | excess vs truth | excess vs ceiling | alpha* | temp recovers | w* | floor recovers | affine recovers |
|---|---|---|---|---|---|---|---|---|---|
| -1 | 1.2332 | 1.2135 | +0.0153 | +0.0092 | 1.006 | +0.0000 | 0.0001 | +0.0000 | +0.0002 |
| 0 | 1.5726 | 1.5336 | +0.0874 | +0.0767 | 0.950 | +0.0012 | 0.0023 | +0.0001 | +0.0017 |
| 1 | 1.4210 | 1.3537 | +0.0736 | +0.0645 | 0.992 | +0.0000 | 0.0004 | +0.0000 | -0.0003 |
| 2 | 1.3639 | 1.3113 | +0.0580 | +0.0503 | 0.993 | +0.0000 | 0.0002 | +0.0000 | -0.0005 |
| 3 | 1.2642 | 1.2268 | +0.0389 | +0.0317 | 0.998 | +0.0000 | 0.0004 | +0.0000 | -0.0002 |
| 4 | 1.2513 | 1.2188 | +0.0366 | +0.0278 | 0.995 | +0.0000 | 0.0007 | +0.0000 | -0.0001 |
| 5 | 1.2439 | 1.2161 | +0.0438 | +0.0349 | 0.980 | +0.0002 | 0.0021 | +0.0001 | +0.0003 |
| 6 | 1.2635 | 1.2325 | +0.0293 | +0.0202 | 1.002 | +0.0000 | 0.0001 | +0.0000 | +0.0000 |
| 7 | 1.2345 | 1.2048 | +0.0262 | +0.0174 | 1.005 | +0.0000 | 0.0001 | +0.0000 | -0.0001 |
| 8 | 1.2252 | 1.1944 | +0.0257 | +0.0167 | 1.007 | +0.0000 | 0.0001 | +0.0000 | +0.0000 |
| quiet | 1.2131 | 1.1920 | +0.0154 | +0.0089 | 1.008 | +0.0000 | 0.0001 | +0.0000 | - |

Prize over tau = 0..8: **0.0378 nats per prediction** (0.340 per event); quiet background 0.0089. Fraction recovered:

| consumer | temperature | noise floor |
|---|---|---|
| one knob for the whole stream | -0.001 | +0.000 |
| one knob per post-event offset | +0.004 | +0.001 |
| one knob per event (oracle) | +0.080 | +0.022 |
| one knob per prediction (oracle) | +0.324 | +0.110 |

Mixing the forecast toward observer k at tau = 0 (diagnostic; the model does not have these at inference, and mixing toward `p_kappa*` reaches the ceiling by construction):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| optimal w | 0.003 | 0.000 | 0.000 | 0.130 | 0.617 | 0.998 | 0.999 |
| nats recovered | +0.0001 | +0.0000 | +0.0000 | +0.0035 | +0.0332 | +0.0789 | +0.0874 |

### eps-trained 24k (kappa*_eps = 4.50, 1577 isolated events, H = 8)

| tau | H(q) | H(p_L^eps) | excess vs truth | excess vs ceiling | alpha* | temp recovers | w* | floor recovers | affine recovers |
|---|---|---|---|---|---|---|---|---|---|
| -1 | 1.2673 | 1.2135 | +0.0398 | +0.0262 | 1.019 | +0.0001 | 0.0000 | +0.0000 | +0.0008 |
| 0 | 1.6033 | 1.5336 | +0.1828 | +0.1594 | 0.891 | +0.0065 | 0.0177 | +0.0021 | +0.0095 |
| 1 | 1.4616 | 1.3537 | +0.1461 | +0.1267 | 0.957 | +0.0008 | 0.0038 | +0.0002 | -0.0002 |
| 2 | 1.4147 | 1.3113 | +0.1253 | +0.1094 | 0.975 | +0.0003 | 0.0014 | +0.0000 | -0.0000 |
| 3 | 1.3116 | 1.2268 | +0.0785 | +0.0634 | 1.008 | +0.0000 | 0.0003 | +0.0000 | -0.0008 |
| 4 | 1.3037 | 1.2188 | +0.0863 | +0.0679 | 0.998 | +0.0000 | 0.0008 | +0.0000 | +0.0001 |
| 5 | 1.2981 | 1.2161 | +0.0836 | +0.0646 | 0.998 | +0.0000 | 0.0008 | +0.0000 | +0.0002 |
| 6 | 1.3068 | 1.2325 | +0.0674 | +0.0504 | 1.009 | +0.0000 | 0.0002 | +0.0000 | +0.0001 |
| 7 | 1.2847 | 1.2048 | +0.0673 | +0.0497 | 1.017 | +0.0001 | 0.0002 | +0.0000 | +0.0010 |
| 8 | 1.2758 | 1.1944 | +0.0669 | +0.0495 | 1.019 | +0.0001 | 0.0000 | -0.0000 | +0.0001 |
| quiet | 1.2451 | 1.1920 | +0.0413 | +0.0274 | 1.016 | +0.0001 | 0.0001 | +0.0000 | - |

Prize over tau = 0..8: **0.0823 nats per prediction** (0.741 per event); quiet background 0.0274. Fraction recovered:

| consumer | temperature | noise floor |
|---|---|---|
| one knob for the whole stream | -0.002 | +0.000 |
| one knob per post-event offset | +0.011 | +0.003 |
| one knob per event (oracle) | +0.083 | +0.027 |
| one knob per prediction (oracle) | +0.322 | +0.136 |

Mixing the forecast toward observer k at tau = 0 (diagnostic; the model does not have these at inference, and mixing toward `p_kappa*` reaches the ceiling by construction):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| optimal w | 0.022 | 0.009 | 0.022 | 0.414 | 0.924 | 0.999 | 0.999 |
| nats recovered | +0.0030 | +0.0004 | +0.0005 | +0.0359 | +0.1171 | +0.1743 | +0.1828 |

### eps-trained 8k (kappa*_eps = 3.60, 1577 isolated events, H = 8)

| tau | H(q) | H(p_L^eps) | excess vs truth | excess vs ceiling | alpha* | temp recovers | w* | floor recovers | affine recovers |
|---|---|---|---|---|---|---|---|---|---|
| -1 | 1.3173 | 1.2135 | +0.0904 | +0.0399 | 1.017 | +0.0001 | 0.0006 | +0.0000 | +0.0005 |
| 0 | 1.6129 | 1.5336 | +0.3374 | +0.2470 | 0.786 | +0.0305 | 0.0614 | +0.0149 | +0.0359 |
| 1 | 1.5173 | 1.3537 | +0.2574 | +0.1894 | 0.900 | +0.0049 | 0.0113 | +0.0012 | +0.0038 |
| 2 | 1.4676 | 1.3113 | +0.2272 | +0.1693 | 0.925 | +0.0028 | 0.0040 | +0.0002 | +0.0026 |
| 3 | 1.3601 | 1.2268 | +0.1649 | +0.1099 | 0.963 | +0.0006 | 0.0028 | +0.0002 | -0.0007 |
| 4 | 1.3638 | 1.2188 | +0.1663 | +0.1006 | 0.974 | +0.0003 | 0.0028 | +0.0002 | +0.0001 |
| 5 | 1.3601 | 1.2161 | +0.1599 | +0.0891 | 0.980 | +0.0002 | 0.0020 | +0.0001 | +0.0002 |
| 6 | 1.3540 | 1.2325 | +0.1532 | +0.0963 | 0.962 | +0.0006 | 0.0023 | +0.0001 | +0.0001 |
| 7 | 1.3397 | 1.2048 | +0.1488 | +0.0906 | 0.983 | +0.0001 | 0.0017 | +0.0001 | -0.0005 |
| 8 | 1.3375 | 1.1944 | +0.1488 | +0.0892 | 0.993 | +0.0000 | 0.0010 | +0.0000 | +0.0002 |
| quiet | 1.2945 | 1.1920 | +0.0894 | +0.0391 | 1.017 | +0.0001 | 0.0005 | +0.0000 | - |

Prize over tau = 0..8: **0.1313 nats per prediction** (1.181 per event); quiet background 0.0391. Fraction recovered:

| consumer | temperature | noise floor |
|---|---|---|
| one knob for the whole stream | -0.001 | +0.001 |
| one knob per post-event offset | +0.034 | +0.014 |
| one knob per event (oracle) | +0.112 | +0.041 |
| one knob per prediction (oracle) | +0.414 | +0.198 |

Mixing the forecast toward observer k at tau = 0 (diagnostic; the model does not have these at inference, and mixing toward `p_kappa*` reaches the ceiling by construction):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| optimal w | 0.071 | 0.029 | 0.160 | 0.806 | 0.997 | 0.999 | 0.999 |
| nats recovered | +0.0180 | +0.0021 | +0.0098 | +0.1421 | +0.2705 | +0.3289 | +0.3374 |

### clean-trained 64k (kappa*_eps = 4.85, 1577 isolated events, H = 8)

| tau | H(q) | H(p_L^eps) | excess vs truth | excess vs ceiling | alpha* | temp recovers | w* | floor recovers | affine recovers |
|---|---|---|---|---|---|---|---|---|---|
| -1 | 1.1661 | 1.2135 | +0.0521 | +0.0459 | 0.823 | +0.0094 | 0.0151 | +0.0297 | +0.0092 |
| 0 | 1.1809 | 1.5336 | +1.6925 | +1.6818 | 0.298 | +1.0798 | 0.3641 | +1.1038 | +1.1289 |
| 1 | 1.1816 | 1.3537 | +1.1593 | +1.1502 | 0.372 | +0.6038 | 0.2647 | +0.5363 | +0.5721 |
| 2 | 1.2185 | 1.3113 | +0.7108 | +0.7030 | 0.456 | +0.2968 | 0.1479 | +0.2373 | +0.2802 |
| 3 | 1.1952 | 1.2268 | +0.3137 | +0.3064 | 0.603 | +0.0830 | 0.0537 | +0.0732 | +0.0888 |
| 4 | 1.1880 | 1.2188 | +0.2754 | +0.2666 | 0.626 | +0.0682 | 0.0454 | +0.0641 | +0.0653 |
| 5 | 1.1932 | 1.2161 | +0.2489 | +0.2399 | 0.645 | +0.0570 | 0.0393 | +0.0563 | +0.0434 |
| 6 | 1.2063 | 1.2325 | +0.1906 | +0.1815 | 0.691 | +0.0383 | 0.0299 | +0.0437 | +0.0407 |
| 7 | 1.1802 | 1.2048 | +0.1720 | +0.1632 | 0.708 | +0.0326 | 0.0260 | +0.0399 | +0.0388 |
| 8 | 1.1812 | 1.1944 | +0.1541 | +0.1451 | 0.737 | +0.0245 | 0.0232 | +0.0374 | +0.0248 |
| quiet | 1.1492 | 1.1920 | +0.0554 | +0.0489 | 0.823 | +0.0093 | 0.0150 | +0.0297 | - |

Prize over tau = 0..8: **0.5375 nats per prediction** (4.838 per event); quiet background 0.0489. Fraction recovered:

| consumer | temperature | noise floor |
|---|---|---|
| one knob for the whole stream | +0.241 | +0.309 |
| one knob per post-event offset | +0.472 | +0.453 |
| one knob per event (oracle) | +0.469 | +0.447 |
| one knob per prediction (oracle) | +0.687 | +0.642 |

Mixing the forecast toward observer k at tau = 0 (diagnostic; the model does not have these at inference, and mixing toward `p_kappa*` reaches the ceiling by construction):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| optimal w | 0.376 | 0.568 | 0.769 | 0.920 | 0.998 | 0.999 | 0.999 |
| nats recovered | +1.1338 | +0.8941 | +1.2287 | +1.4938 | +1.6257 | +1.6841 | +1.6925 |

## Q2. The self-supervised readout of the model's own excess surprise

### eps-trained 64k

Target: sum over tau=0..8 of (NLL - H(q)); mean -0.0213, sd 2.613 nats, correlation with the raw future NLL +0.867. 3,670,016 training positions, none labelled. Best block post_block7.

| readout | val R2 | rank corr | AUC(event vs quiet) | hard vs overconfident-wrong | same-prefix pairs (paired) |
|---|---|---|---|---|---|
| state_excess_ridge | +0.000060 | +0.0129 | 0.462 | 0.499 | 0.512 |
| state_excess_mlp | -0.000423 | +0.0131 | 0.522 | 0.529 | 0.480 |
| state_rawnll_ridge | n/a | +0.0025 | 0.550 | 0.048 | 0.463 |
| state_rawnll_mlp | n/a | +0.0024 | 0.595 | 0.038 | 0.417 |
| embed_excess_ridge | -0.000051 | +0.0009 | 0.516 | 0.421 | 0.504 |
| rand_excess_ridge | -0.000044 | +0.0050 | 0.492 | 0.462 | 0.483 |
| rand_rawnll_ridge | n/a | -0.0007 | 0.481 | 0.204 | nan |
| Hsum_raw | n/a | +0.0177 | 0.668 | 0.000 | nan |
| Hq_raw | -0.374867 | +0.0043 | 0.627 | 0.077 | nan |
| H_next (reference) | - | - | - | - | 0.440 |
| oracle_linear (reference) | - | - | - | - | 0.717 |
| oracle_mlp (reference) | - | - | - | - | 0.885 |

Pairs: 8243 matched, 1570 held-out; prefix identity check 4.3e-06. `oracle_*` are probes on the SAME states trained with the violation label.

| tau | -2 | -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| state_excess_ridge | -0.022 | -0.022 | -0.028 | -0.029 | -0.026 | -0.023 | -0.023 | -0.025 | -0.025 | -0.027 | -0.025 |
| state_excess_mlp | -0.022 | -0.023 | -0.020 | -0.005 | -0.014 | -0.020 | -0.016 | -0.023 | -0.023 | -0.027 | -0.023 |
| rand_excess_ridge | -0.023 | -0.023 | -0.025 | -0.025 | -0.025 | -0.026 | -0.026 | -0.026 | -0.026 | -0.027 | -0.027 |

### eps-trained 8k

Target: sum over tau=0..8 of (NLL - H(q)); mean -0.0237, sd 2.664 nats, correlation with the raw future NLL +0.879. 3,670,016 training positions, none labelled. Best block post_block3.

| readout | val R2 | rank corr | AUC(event vs quiet) | hard vs overconfident-wrong | same-prefix pairs (paired) |
|---|---|---|---|---|---|
| state_excess_ridge | +0.000374 | +0.0245 | 0.583 | 0.545 | 0.476 |
| state_excess_mlp | -0.003223 | +0.0218 | 0.567 | 0.533 | 0.506 |
| state_rawnll_ridge | n/a | +0.0031 | 0.558 | 0.049 | 0.489 |
| state_rawnll_mlp | n/a | +0.0053 | 0.595 | 0.049 | 0.400 |
| embed_excess_ridge | +0.000105 | +0.0142 | 0.541 | 0.639 | 0.499 |
| rand_excess_ridge | +0.000315 | +0.0206 | 0.508 | 0.588 | 0.500 |
| rand_rawnll_ridge | n/a | -0.0057 | 0.473 | 0.222 | nan |
| Hsum_raw | n/a | +0.0428 | 0.691 | 0.000 | nan |
| Hq_raw | -0.384699 | +0.0062 | 0.618 | 0.065 | nan |
| H_next (reference) | - | - | - | - | 0.420 |
| oracle_linear (reference) | - | - | - | - | 0.665 |
| oracle_mlp (reference) | - | - | - | - | 0.786 |

Pairs: 10563 matched, 2305 held-out; prefix identity check 4.3e-06. `oracle_*` are probes on the SAME states trained with the violation label.

| tau | -2 | -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| state_excess_ridge | -0.023 | -0.022 | -0.007 | +0.006 | -0.002 | -0.010 | -0.012 | -0.015 | -0.016 | -0.018 | -0.017 |
| state_excess_mlp | -0.027 | -0.020 | +0.014 | +0.031 | +0.025 | +0.008 | +0.012 | -0.000 | -0.009 | -0.012 | -0.029 |
| rand_excess_ridge | -0.021 | -0.021 | -0.019 | -0.019 | -0.020 | -0.020 | -0.019 | -0.020 | -0.021 | -0.021 | -0.022 |

### clean-trained 64k, corrupted stream

Target: sum over tau=0..8 of (NLL - H(q)); mean +1.2829, sd 4.879 nats, correlation with the raw future NLL +0.951. 3,670,016 training positions, none labelled. Best block post_block7.

| readout | val R2 | rank corr | AUC(event vs quiet) | hard vs overconfident-wrong | same-prefix pairs (paired) |
|---|---|---|---|---|---|
| state_excess_ridge | +0.003462 | +0.0413 | 0.832 | 0.724 | 0.449 |
| state_excess_mlp | +0.006363 | +0.0379 | 0.846 | 0.670 | 0.438 |
| state_rawnll_ridge | n/a | -0.0029 | 0.595 | 0.071 | 0.471 |
| state_rawnll_mlp | n/a | +0.0092 | 0.738 | 0.081 | 0.390 |
| embed_excess_ridge | +0.000134 | +0.0081 | 0.592 | 0.680 | 0.498 |
| rand_excess_ridge | +0.000139 | +0.0097 | 0.580 | 0.634 | 0.500 |
| rand_rawnll_ridge | n/a | -0.0098 | 0.489 | 0.219 | nan |
| Hsum_raw | n/a | -0.0799 | 0.563 | 0.000 | nan |
| Hq_raw | -0.041966 | -0.0131 | 0.514 | 0.083 | nan |
| H_next (reference) | - | - | - | - | 0.433 |
| oracle_linear (reference) | - | - | - | - | 0.706 |
| oracle_mlp (reference) | - | - | - | - | 0.862 |

Pairs: 5581 matched, 1181 held-out; prefix identity check 2.4e-05. `oracle_*` are probes on the SAME states trained with the violation label.

| tau | -2 | -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| state_excess_ridge | +1.251 | +1.264 | +1.787 | +1.860 | +1.733 | +1.622 | +1.568 | +1.518 | +1.459 | +1.438 | +1.413 |
| state_excess_mlp | +1.247 | +1.246 | +2.424 | +2.511 | +2.153 | +1.768 | +1.667 | +1.586 | +1.508 | +1.450 | +1.433 |
| rand_excess_ridge | +1.289 | +1.288 | +1.311 | +1.293 | +1.289 | +1.289 | +1.290 | +1.289 | +1.293 | +1.288 | +1.292 |

## Q3. The gain loop

### eps-trained 64k (prize 0.0403 nats/prediction over 1825 events; 0.0378 over the 1577 isolated ones)

Maps are fit on held-out corrupted windows by the model's own realised NLL, at window positions t >= 8 only -- the positions they are applied at. A negative cost is an improvement.

| arm | alpha at tau=0 | prize recovered (vs the eps-observer ceiling) | d realised NLL, event window | d realised NLL, quiet | d realised NLL, whole stream | H(q) after event |
|---|---|---|---|---|---|---|
| base | 1.000 | +0.000 | +0.0000 | +0.00000 | +0.00000 | 1.5774 |
| const | 1.004 | -0.001 | +0.0008 | -0.00028 | -0.00001 | 1.5741 |
| head_ridge | 1.005 | -0.002 | +0.0008 | -0.00040 | -0.00006 | 1.5736 |
| head_ridge_loglin | 1.006 | -0.001 | +0.0010 | -0.00051 | -0.00008 | 1.5726 |
| head_mlp | 1.003 | -0.000 | +0.0008 | -0.00038 | -0.00006 | 1.5742 |
| head_mlp_loglin | 1.004 | -0.001 | +0.0010 | -0.00050 | -0.00005 | 1.5733 |
| oracle_probe | 0.998 | +0.000 | +0.0005 | -0.00024 | -0.00002 | 1.5788 |
| oracle_offset | 0.342 | -1.105 | -0.0808 | -0.00196 | -0.00705 | 2.5387 |

The two loss columns can disagree in sign, and the reason is not noise. The post-event population is selected on a latent -- whether the token at t was corrupted -- that no observer of the window can see, so the conditional law of the realised token there is NOT `p_L^eps`. Against the ceiling a window-observer could actually reach, the prize column is the honest one; on realised tokens given side information about the latent, the NLL columns are.

Same-prefix pairs (1570 held-out), paired win rate that the forecast after the ILLEGAL token is LESS confident than after its legal twin (> 0.5 = the normative direction):

| arm | H_next paired | alpha viol / twin |
|---|---|---|
| base | 0.440 | 1.000 / 1.000 |
| const | 0.440 | 1.004 / 1.004 |
| oracle_probe | 0.441 | 1.001 / 1.001 |
| head_ridge | 0.434 | 1.005 / 1.005 |
| head_ridge_loglin | 0.435 | 1.006 / 1.007 |
| head_mlp | 0.433 | 1.005 / 1.005 |
| head_mlp_loglin | 0.438 | 1.006 / 1.005 |

Fitted maps (alpha per quantile bin of the gate, fit by the model's own realised NLL on held-out windows):

- `head_ridge_map`: [1.0169, 0.9911, 1.0789, 1.0454, 1.0441, 1.0269, 1.017, 1.0083, 1.0051, 1.0029, 0.996, 0.9859, 0.9792, 0.9726, 0.9691, 0.9917, 1.0143, 0.9708] (monotone increasing False, decreasing False)
- `head_mlp_map`: [1.0301, 1.0714, 1.0183, 1.0579, 1.0247, 1.022, 1.0207, 1.0086, 1.0068, 1.0013, 0.9941, 0.9838, 0.9893, 0.9871, 0.9959, 1.0114, 1.0297, 0.9827] (monotone increasing False, decreasing False)
- `oracle_probe_map`: [1.0001, 0.9844, 1.0295, 0.998, 0.991, 0.9973, 1.0053, 1.0107, 1.0047, 1.0028, 1.0016, 0.9958, 0.9984, 1.009, 0.9933, 0.9871, 0.9943, 0.9914] (monotone increasing False, decreasing False)
- `oracle_offset`: {'0': 0.34212478860682594, '1': 0.8912451469860327, '2': 0.958442475562073, '3': 0.9825970883632308, '4': 1.0084259760409182, '5': 1.006236821230671, '6': 1.0131192079188072, '7': 1.0061157736000674, '8': 1.026952454385504, '9': 1.0284739445168287} (index = offset since the last corrupted token; 9 = none within H)
- constant: 1.0036; oracle corruption probe AUC 0.814

### clean-trained 64k, corrupted stream (prize 0.5764 nats/prediction over 1825 events; 0.5375 over the 1577 isolated ones)

Maps are fit on held-out corrupted windows by the model's own realised NLL, at window positions t >= 8 only -- the positions they are applied at. A negative cost is an improvement.

| arm | alpha at tau=0 | prize recovered (vs the eps-observer ceiling) | d realised NLL, event window | d realised NLL, quiet | d realised NLL, whole stream | H(q) after event |
|---|---|---|---|---|---|---|
| base | 1.000 | +0.000 | +0.0000 | +0.00000 | +0.00000 | 1.1771 |
| const | 0.764 | +0.248 | -0.2407 | +0.01887 | -0.01726 | 1.3199 |
| head_ridge | 0.593 | +0.431 | -0.3740 | +0.01464 | -0.02995 | 1.5312 |
| head_ridge_loglin | 0.605 | +0.426 | -0.3732 | +0.01607 | -0.02940 | 1.5038 |
| head_mlp | 0.568 | +0.486 | -0.4056 | +0.01285 | -0.03159 | 1.6591 |
| head_mlp_loglin | 0.573 | +0.484 | -0.4072 | +0.01286 | -0.03125 | 1.6324 |
| oracle_probe | 0.502 | +0.427 | -0.3697 | +0.01597 | -0.02569 | 1.7237 |
| oracle_offset | 0.068 | +0.406 | -0.6177 | +0.00587 | -0.05096 | 2.7442 |

The two loss columns can disagree in sign, and the reason is not noise. The post-event population is selected on a latent -- whether the token at t was corrupted -- that no observer of the window can see, so the conditional law of the realised token there is NOT `p_L^eps`. Against the ceiling a window-observer could actually reach, the prize column is the honest one; on realised tokens given side information about the latent, the NLL columns are.

Same-prefix pairs (1181 held-out), paired win rate that the forecast after the ILLEGAL token is LESS confident than after its legal twin (> 0.5 = the normative direction):

| arm | H_next paired | alpha viol / twin |
|---|---|---|
| base | 0.433 | 1.000 / 1.000 |
| const | 0.434 | 0.764 / 0.764 |
| oracle_probe | 0.390 | 0.706 / 0.672 |
| head_ridge | 0.395 | 0.694 / 0.672 |
| head_ridge_loglin | 0.401 | 0.683 / 0.667 |
| head_mlp | 0.379 | 0.666 / 0.629 |
| head_mlp_loglin | 0.378 | 0.661 / 0.630 |

Fitted maps (alpha per quantile bin of the gate, fit by the model's own realised NLL on held-out windows):

- `head_ridge_map`: [1.0148, 0.9934, 0.9552, 0.9922, 0.9745, 0.9604, 0.9392, 0.9067, 0.8569, 0.7884, 0.6966, 0.59, 0.5057, 0.4265, 0.3817, 0.3199, 0.2181, 0.1674] (monotone increasing False, decreasing False)
- `head_mlp_map`: [0.8624, 0.8776, 0.8776, 0.8898, 0.8834, 0.8764, 0.8893, 0.8852, 0.8665, 0.8196, 0.7416, 0.6468, 0.5463, 0.4239, 0.3378, 0.2699, 0.2167, 0.1651] (monotone increasing False, decreasing False)
- `oracle_probe_map`: [1.0041, 1.0246, 0.9345, 0.9649, 0.9351, 0.9027, 0.8897, 0.8722, 0.8354, 0.7946, 0.7444, 0.6901, 0.5928, 0.4621, 0.3387, 0.2351, 0.166, 0.1338] (monotone increasing False, decreasing False)
- `oracle_offset`: {'0': 0.06834764469368362, '1': 0.35175598136425346, '2': 0.4462158115631737, '3': 0.5778267303416599, '4': 0.631311612314197, '5': 0.6855641254717784, '6': 0.7066005129539874, '7': 0.7459975048433577, '8': 0.7745964073653782, '9': 0.8746012647087598} (index = offset since the last corrupted token; 9 = none within H)
- constant: 0.7640; oracle corruption probe AUC 0.873

## Q4. The plasticity loop

### step 12000 (continued from eps-trained step 8000)

| arm | kappa* (eps, corrupted) | KL(p_kappa*^eps||q) | CE corrupted | CE clean | KL(p_L||q) clean | best_k clean | event excess vs ceiling | quiet excess |
|---|---|---|---|---|---|---|---|---|
| head_down | 3.75 | 0.0499 | 1.3412 | - | - | - | - | - |
| head_shuf | 3.70 | 0.0495 | 1.3414 | - | - | - | - | - |
| oracle_down | 3.75 | 0.0541 | 1.3451 | - | - | - | - | - |
| oracle_up | 3.70 | 0.0488 | 1.3424 | - | - | - | - | - |
| uniform | 3.75 | 0.0461 | 1.3350 | - | - | - | - | - |

### step 24000 (continued from eps-trained step 8000)

| arm | kappa* (eps, corrupted) | KL(p_kappa*^eps||q) | CE corrupted | CE clean | KL(p_L||q) clean | best_k clean | event excess vs ceiling | quiet excess |
|---|---|---|---|---|---|---|---|---|
| head_down | 4.45 | 0.0358 | 1.3136 | 1.2448 | 0.0554 | 4 | 0.0929 | 0.0305 |
| head_shuf | 4.45 | 0.0346 | 1.3130 | 1.2444 | 0.0551 | 4 | 0.0885 | 0.0295 |
| oracle_down | 4.45 | 0.0403 | 1.3181 | 1.2393 | 0.0500 | 4 | 0.1478 | 0.0304 |
| oracle_up | 4.45 | 0.0355 | 1.3146 | 1.2497 | 0.0604 | 4 | 0.0799 | 0.0333 |
| uniform | 4.50 | 0.0310 | 1.3077 | 1.2396 | 0.0502 | 4 | 0.0801 | 0.0255 |

Distance to each clean observer, `KL(p_k||q)` on clean windows:

| arm | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|---|
| head_down | 2.1883 | 1.0571 | 0.3509 | 0.1062 | 0.0475 | 0.0503 | 0.0554 |
| head_shuf | 2.1946 | 1.0418 | 0.3401 | 0.1022 | 0.0460 | 0.0499 | 0.0551 |
| oracle_down | 2.3665 | 1.1215 | 0.3622 | 0.1055 | 0.0424 | 0.0448 | 0.0500 |
| oracle_up | 2.0946 | 0.9976 | 0.3317 | 0.1029 | 0.0501 | 0.0551 | 0.0604 |
| uniform | 2.2021 | 1.0526 | 0.3512 | 0.1082 | 0.0443 | 0.0452 | 0.0502 |

Per-leaf-level CE on clean windows:

| arm | lev 0 | lev 1 | lev 2 | lev 3 | lev 4 | lev 5 | lev 6 |
|---|---|---|---|---|---|---|---|
| head_down | 0.5168 | 1.6352 | 2.1379 | 2.4383 | 2.5242 | 2.5273 | 2.5316 |
| head_shuf | 0.5161 | 1.6347 | 2.1396 | 2.4386 | 2.5236 | 2.5259 | 2.5299 |
| oracle_down | 0.5087 | 1.6317 | 2.1352 | 2.4369 | 2.5213 | 2.5251 | 2.5300 |
| oracle_up | 0.5238 | 1.6386 | 2.1423 | 2.4395 | 2.5239 | 2.5261 | 2.5296 |
| uniform | 0.5128 | 1.6286 | 2.1301 | 2.4333 | 2.5204 | 2.5236 | 2.5276 |

Gate diagnostics during training (correlation of the gate with the realised horizon excess, and the unweighted train loss):

| arm | gate-target corr (first / last) | train loss (first / last) |
|---|---|---|
| head_down | +0.0286 / -0.0374 | 1.3724 / 1.3149 |
| head_shuf | +0.0286 / -0.0350 | 1.3724 / 1.3146 |
| oracle_down | +0.0135 / +0.0014 | 1.3724 / 1.3202 |
| oracle_up | +0.0135 / -0.0123 | 1.3724 / 1.3162 |
| uniform | +nan / +nan | 1.3724 / 1.3089 |

