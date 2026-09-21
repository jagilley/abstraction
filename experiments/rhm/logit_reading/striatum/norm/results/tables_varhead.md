# norm -- the mean-and-variance critic (2026-09-17)

Facts only. Xiang, Lohrenz & Montague's insula row is a **variance prediction error**, which a mean-only ridge has no analogue for. On a 0/1 outcome a variance head is degenerate (`Var = V(1 - V)`, a function of the mean), so this needs a continuous target. `norm/task.py`'s `var_target` knob (default off, so every banked cell is untouched) fits, on the SAME cached states, the same lambda ladder and the same val rows as the banked critic:

- a **mean head** `Vc[l, d]` on a continuous target, and

- a **variance head** `Vvar[l, d]` fitted on that head's own squared training residual `(y - Vc)^2`.

Then at the event, one token before it, `dc = y - Vc_pre` is the outcome surprise in the continuous currency and **`vpe = dc^2 - Vvar_pre`** is the variance prediction error.

**The target.** `lmean`: `y(t) = mean over l = 1..4 of o_l(t)`, the actor being right at each of the four absorbed levels at one position, in {0, .25, .5, .75, 1}. (`hsum`, the horizon sum over `delta = 0..8`, was tried first and fits far worse -- mean-head held-out R^2 0.03-0.06 against `lmean`'s 0.20-0.27 -- so it is not carried here.)

**Degeneracy gate.** `R2 var | quad(mean)` is the held-out R^2 of the best quadratic in the mean head's own output, fitted on the same training rows and scored on the same val rows. If the variance head does not beat it, there is no variance signal in the state beyond the mean and the column set is not worth reading.

Reproduction:

```bash
cd experiments   # MODAL_PROFILE=chromatic
D=/data/v16_s2_L6_m4_distinct/logit_reading
modal run -m rhm.logit_reading.striatum.norm.task::norm_sweep \
    --cells "$D/traj_a1_s42/step064000.pt:swap65k:0,$D/traj_a1_s42/step000000.pt:swap65k:0,$D/traj_a1_s42/step064000.pt:a1:1" \
    --var-target lmean --max-twins 0 --tag var
python -m rhm.logit_reading.striatum.norm.analyze <dir>/traj_a1_s42 \
    --mode varhead --addendum-tag var --outfile tables_varhead.md
```

## venue `a1`

### The two heads' held-out fits

| step  | target | col   | E[y]   | sd(y)  | R2 mean | E[(y-Vc)^2] | R2 var | R2 var | quad(mean) |
|-------|--------|-------|--------|--------|---------|-------------|--------|---------------------|
| 64000 | lmean  | l0_d0 | 0.6859 | 0.2960 | 0.2702  | 0.0639      | 0.0978 | 0.0623              |
| 64000 | lmean  | l0_d1 | 0.6886 | 0.2961 | 0.2285  | 0.0676      | 0.0704 | 0.0456              |

### step 64000, anchor the edit onset (`first_diff`)

**`l0`** -- 4097 held-out rows. `E[dc]` -0.1263, `E[dc^2]` 0.1116, `E[Vvar_pre]` 0.0875, `E[vpe]` 0.0241; corr(Vvar_pre, dc^2) = -0.0903.

Binned by the SIGNED outcome surprise `dc` (7 equal-count bins):

| bin | n   | mean dc | mean dc^2 | mean Vvar_pre | mean vpe | mean dc^2 - E[dc^2] |
|-----|-----|---------|-----------|---------------|----------|---------------------|
| 1   | 586 | -0.6051 | 0.3795    | 0.0830        | 0.2965   | 0.2678              |
| 2   | 585 | -0.3897 | 0.1535    | 0.0889        | 0.0646   | 0.0419              |
| 3   | 585 | -0.2493 | 0.0638    | 0.0894        | -0.0255  | -0.0478             |
| 4   | 586 | -0.1297 | 0.0177    | 0.0868        | -0.0691  | -0.0939             |
| 5   | 585 | -0.0079 | 0.0018    | 0.0877        | -0.0859  | -0.1098             |
| 6   | 585 | 0.1355  | 0.0201    | 0.0836        | -0.0635  | -0.0915             |
| 7   | 585 | 0.3629  | 0.1446    | 0.0935        | 0.0511   | 0.0330              |

U statistic (outer two bins minus the middle bin): **vpe 0.2430**, constant-variance baseline 0.2444, `Vvar_pre` 0.0014.

Binned by `|dc|` (5 equal-count bins) -- the ordering Xiang's insula row is read by:

| bin | n   | mean |dc| | mean Vvar_pre | mean vpe |
|-----|-----|-----------|---------------|----------|
| 1   | 820 | 0.0505    | 0.0876        | -0.0843  |
| 2   | 819 | 0.1375    | 0.0846        | -0.0650  |
| 3   | 820 | 0.2358    | 0.0887        | -0.0319  |
| 4   | 819 | 0.3660    | 0.0903        | 0.0452   |
| 5   | 819 | 0.5744    | 0.0865        | 0.2566   |

The response at the event, by `etype` (`Rc` is the mean head's revision, `Rvar` the variance head's -- does the critic's expected squared error RISE when the violation arrives):

| etype           | n    | mean Rc | mean Rvar | mean Vvar_pre |
|-----------------|------|---------|-----------|---------------|
| all             | 4097 | -0.0207 | -0.0012   | 0.0875        |
| swap            | 2021 | -0.0269 | -0.0006   | 0.0869        |
| rare            | 1029 | -0.0165 | -0.0013   | 0.0901        |
| none (unedited) | 1047 | -0.0130 | -0.0023   | 0.0863        |

### step 64000, anchor `t_v`

**`l0`** -- 1659 held-out rows. `E[dc]` -0.2399, `E[dc^2]` 0.1457, `E[Vvar_pre]` 0.0620, `E[vpe]` 0.0838; corr(Vvar_pre, dc^2) = -0.0548.

Binned by the SIGNED outcome surprise `dc` (7 equal-count bins):

| bin | n   | mean dc | mean dc^2 | mean Vvar_pre | mean vpe | mean dc^2 - E[dc^2] |
|-----|-----|---------|-----------|---------------|----------|---------------------|
| 1   | 237 | -0.7106 | 0.5138    | 0.0608        | 0.4530   | 0.3680              |
| 2   | 237 | -0.5009 | 0.2524    | 0.0624        | 0.1899   | 0.1066              |
| 3   | 237 | -0.3513 | 0.1253    | 0.0608        | 0.0645   | -0.0205             |
| 4   | 237 | -0.2403 | 0.0588    | 0.0610        | -0.0023  | -0.0870             |
| 5   | 237 | -0.1124 | 0.0142    | 0.0602        | -0.0460  | -0.1315             |
| 6   | 237 | 0.0206  | 0.0019    | 0.0645        | -0.0626  | -0.1438             |
| 7   | 237 | 0.2154  | 0.0539    | 0.0641        | -0.0102  | -0.0919             |

U statistic (outer two bins minus the middle bin): **vpe 0.2236**, constant-variance baseline 0.2251, `Vvar_pre` 0.0014.

Binned by `|dc|` (5 equal-count bins) -- the ordering Xiang's insula row is read by:

| bin | n   | mean |dc| | mean Vvar_pre | mean vpe |
|-----|-----|-----------|---------------|----------|
| 1   | 332 | 0.0460    | 0.0631        | -0.0602  |
| 2   | 332 | 0.1552    | 0.0610        | -0.0356  |
| 3   | 332 | 0.2674    | 0.0606        | 0.0119   |
| 4   | 332 | 0.4177    | 0.0646        | 0.1133   |
| 5   | 331 | 0.6623    | 0.0606        | 0.3903   |

The response at the event, by `etype` (`Rc` is the mean head's revision, `Rvar` the variance head's -- does the critic's expected squared error RISE when the violation arrives):

| etype | n    | mean Rc | mean Rvar | mean Vvar_pre |
|-------|------|---------|-----------|---------------|
| all   | 1659 | -0.0460 | 0.0009    | 0.0620        |
| swap  | 1659 | -0.0460 | 0.0009    | 0.0620        |

## venue `swap65k`

### The two heads' held-out fits

| step  | target | col   | E[y]   | sd(y)  | R2 mean | E[(y-Vc)^2] | R2 var | R2 var | quad(mean) |
|-------|--------|-------|--------|--------|---------|-------------|--------|---------------------|
| 0     | lmean  | l0_d0 | 0.2681 | 0.2314 | 0.0542  | 0.0506      | 0.0057 | 0.0020              |
| 0     | lmean  | l0_d1 | 0.2659 | 0.2307 | 0.0399  | 0.0511      | 0.0045 | 0.0013              |
| 64000 | lmean  | l0_d0 | 0.6509 | 0.3008 | 0.2578  | 0.0672      | 0.0910 | 0.0541              |
| 64000 | lmean  | l0_d1 | 0.6530 | 0.3011 | 0.2112  | 0.0715      | 0.0581 | 0.0361              |

### step 0, anchor the edit onset (`first_diff`)

**`l0`** -- 16373 held-out rows. `E[dc]` -0.0877, `E[dc^2]` 0.0459, `E[Vvar_pre]` 0.0516, `E[vpe]` -0.0057; corr(Vvar_pre, dc^2) = 0.0786.

Binned by the SIGNED outcome surprise `dc` (7 equal-count bins):

| bin | n    | mean dc | mean dc^2 | mean Vvar_pre | mean vpe | mean dc^2 - E[dc^2] |
|-----|------|---------|-----------|---------------|----------|---------------------|
| 1   | 2339 | -0.2993 | 0.0900    | 0.0547        | 0.0354   | 0.0441              |
| 2   | 2339 | -0.2584 | 0.0668    | 0.0530        | 0.0138   | 0.0209              |
| 3   | 2339 | -0.2320 | 0.0539    | 0.0495        | 0.0044   | 0.0080              |
| 4   | 2339 | -0.1205 | 0.0201    | 0.0507        | -0.0306  | -0.0258             |
| 5   | 2339 | -0.0121 | 0.0003    | 0.0535        | -0.0532  | -0.0456             |
| 6   | 2339 | 0.0273  | 0.0010    | 0.0481        | -0.0471  | -0.0449             |
| 7   | 2339 | 0.2812  | 0.0893    | 0.0521        | 0.0372   | 0.0434              |

U statistic (outer two bins minus the middle bin): **vpe 0.0669**, constant-variance baseline 0.0696, `Vvar_pre` 0.0027.

Binned by `|dc|` (5 equal-count bins) -- the ordering Xiang's insula row is read by:

| bin | n    | mean |dc| | mean Vvar_pre | mean vpe |
|-----|------|-----------|---------------|----------|
| 1   | 3275 | 0.0124    | 0.0519        | -0.0517  |
| 2   | 3275 | 0.0706    | 0.0508        | -0.0428  |
| 3   | 3274 | 0.2240    | 0.0497        | 0.0006   |
| 4   | 3275 | 0.2552    | 0.0523        | 0.0129   |
| 5   | 3274 | 0.3177    | 0.0534        | 0.0525   |

The response at the event, by `etype` (`Rc` is the mean head's revision, `Rvar` the variance head's -- does the critic's expected squared error RISE when the violation arrives):

| etype | n     | mean Rc | mean Rvar | mean Vvar_pre |
|-------|-------|---------|-----------|---------------|
| all   | 16373 | -0.0022 | -0.0023   | 0.0516        |
| swap  | 16373 | -0.0022 | -0.0023   | 0.0516        |

### step 0, anchor `t_v`

**`l0`** -- 13428 held-out rows. `E[dc]` -0.0479, `E[dc^2]` 0.0485, `E[Vvar_pre]` 0.0497, `E[vpe]` -0.0012; corr(Vvar_pre, dc^2) = 0.0835.

Binned by the SIGNED outcome surprise `dc` (7 equal-count bins):

| bin | n    | mean dc | mean dc^2 | mean Vvar_pre | mean vpe | mean dc^2 - E[dc^2] |
|-----|------|---------|-----------|---------------|----------|---------------------|
| 1   | 1919 | -0.2942 | 0.0872    | 0.0521        | 0.0351   | 0.0387              |
| 2   | 1918 | -0.2433 | 0.0593    | 0.0507        | 0.0086   | 0.0108              |
| 3   | 1918 | -0.2068 | 0.0431    | 0.0464        | -0.0033  | -0.0054             |
| 4   | 1919 | -0.0479 | 0.0029    | 0.0511        | -0.0483  | -0.0457             |
| 5   | 1918 | 0.0086  | 0.0002    | 0.0500        | -0.0498  | -0.0483             |
| 6   | 1918 | 0.1100  | 0.0182    | 0.0480        | -0.0298  | -0.0303             |
| 7   | 1918 | 0.3383  | 0.1289    | 0.0496        | 0.0793   | 0.0803              |

U statistic (outer two bins minus the middle bin): **vpe 0.1055**, constant-variance baseline 0.1052, `Vvar_pre` -0.0003.

Binned by `|dc|` (5 equal-count bins) -- the ordering Xiang's insula row is read by:

| bin | n    | mean |dc| | mean Vvar_pre | mean vpe |
|-----|------|-----------|---------------|----------|
| 1   | 2686 | 0.0152    | 0.0504        | -0.0501  |
| 2   | 2686 | 0.0728    | 0.0483        | -0.0410  |
| 3   | 2685 | 0.2141    | 0.0480        | -0.0021  |
| 4   | 2686 | 0.2511    | 0.0513        | 0.0119   |
| 5   | 2685 | 0.3416    | 0.0505        | 0.0754   |

The response at the event, by `etype` (`Rc` is the mean head's revision, `Rvar` the variance head's -- does the critic's expected squared error RISE when the violation arrives):

| etype | n     | mean Rc | mean Rvar | mean Vvar_pre |
|-------|-------|---------|-----------|---------------|
| all   | 13428 | 0.0034  | -0.0002   | 0.0497        |
| swap  | 13428 | 0.0034  | -0.0002   | 0.0497        |

### step 64000, anchor the edit onset (`first_diff`)

**`l0`** -- 16373 held-out rows. `E[dc]` -0.1911, `E[dc^2]` 0.1203, `E[Vvar_pre]` 0.0896, `E[vpe]` 0.0306; corr(Vvar_pre, dc^2) = -0.1730.

Binned by the SIGNED outcome surprise `dc` (7 equal-count bins):

| bin | n    | mean dc | mean dc^2 | mean Vvar_pre | mean vpe | mean dc^2 - E[dc^2] |
|-----|------|---------|-----------|---------------|----------|---------------------|
| 1   | 2339 | -0.6291 | 0.4048    | 0.0816        | 0.3231   | 0.2845              |
| 2   | 2339 | -0.4359 | 0.1915    | 0.0885        | 0.1030   | 0.0712              |
| 3   | 2339 | -0.3138 | 0.0997    | 0.0909        | 0.0088   | -0.0206             |
| 4   | 2339 | -0.1957 | 0.0394    | 0.0911        | -0.0517  | -0.0809             |
| 5   | 2339 | -0.0891 | 0.0089    | 0.0907        | -0.0817  | -0.1113             |
| 6   | 2339 | 0.0467  | 0.0045    | 0.0922        | -0.0878  | -0.1158             |
| 7   | 2339 | 0.2796  | 0.0931    | 0.0923        | 0.0008   | -0.0272             |

U statistic (outer two bins minus the middle bin): **vpe 0.2137**, constant-variance baseline 0.2095, `Vvar_pre` -0.0042.

Binned by `|dc|` (5 equal-count bins) -- the ordering Xiang's insula row is read by:

| bin | n    | mean |dc| | mean Vvar_pre | mean vpe |
|-----|------|-----------|---------------|----------|
| 1   | 3275 | 0.0510    | 0.0920        | -0.0885  |
| 2   | 3275 | 0.1447    | 0.0902        | -0.0685  |
| 3   | 3274 | 0.2539    | 0.0920        | -0.0262  |
| 4   | 3275 | 0.3856    | 0.0899        | 0.0603   |
| 5   | 3274 | 0.5913    | 0.0840        | 0.2762   |

The response at the event, by `etype` (`Rc` is the mean head's revision, `Rvar` the variance head's -- does the critic's expected squared error RISE when the violation arrives):

| etype | n     | mean Rc | mean Rvar | mean Vvar_pre |
|-------|-------|---------|-----------|---------------|
| all   | 16373 | -0.0310 | -0.0019   | 0.0896        |
| swap  | 16373 | -0.0310 | -0.0019   | 0.0896        |

### step 64000, anchor `t_v`

**`l0`** -- 13428 held-out rows. `E[dc]` -0.2000, `E[dc^2]` 0.1275, `E[Vvar_pre]` 0.0696, `E[vpe]` 0.0579; corr(Vvar_pre, dc^2) = -0.0575.

Binned by the SIGNED outcome surprise `dc` (7 equal-count bins):

| bin | n    | mean dc | mean dc^2 | mean Vvar_pre | mean vpe | mean dc^2 - E[dc^2] |
|-----|------|---------|-----------|---------------|----------|---------------------|
| 1   | 1919 | -0.6668 | 0.4521    | 0.0683        | 0.3838   | 0.3246              |
| 2   | 1918 | -0.4675 | 0.2201    | 0.0688        | 0.1513   | 0.0926              |
| 3   | 1918 | -0.3143 | 0.1008    | 0.0680        | 0.0328   | -0.0267             |
| 4   | 1919 | -0.1942 | 0.0387    | 0.0700        | -0.0314  | -0.0888             |
| 5   | 1918 | -0.0680 | 0.0064    | 0.0695        | -0.0632  | -0.1212             |
| 6   | 1918 | 0.0612  | 0.0050    | 0.0709        | -0.0658  | -0.1225             |
| 7   | 1918 | 0.2498  | 0.0693    | 0.0716        | -0.0022  | -0.0582             |

U statistic (outer two bins minus the middle bin): **vpe 0.2222**, constant-variance baseline 0.2221, `Vvar_pre` -0.0001.

Binned by `|dc|` (5 equal-count bins) -- the ordering Xiang's insula row is read by:

| bin | n    | mean |dc| | mean Vvar_pre | mean vpe |
|-----|------|-----------|---------------|----------|
| 1   | 2686 | 0.0442    | 0.0689        | -0.0662  |
| 2   | 2686 | 0.1423    | 0.0727        | -0.0515  |
| 3   | 2685 | 0.2455    | 0.0670        | -0.0058  |
| 4   | 2686 | 0.3909    | 0.0717        | 0.0840   |
| 5   | 2685 | 0.6217    | 0.0678        | 0.3293   |

The response at the event, by `etype` (`Rc` is the mean head's revision, `Rvar` the variance head's -- does the critic's expected squared error RISE when the violation arrives):

| etype | n     | mean Rc | mean Rvar | mean Vvar_pre |
|-------|-------|---------|-----------|---------------|
| all   | 13428 | -0.0426 | -0.0021   | 0.0696        |
| swap  | 13428 | -0.0426 | -0.0021   | 0.0696        |

